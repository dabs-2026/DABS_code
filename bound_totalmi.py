import math

import torch
import pyro
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import warn_if_nan
from pyro.contrib.util import lexpand
from pyro.infer.util import torch_item

from bound import (
    MutualInformation,
    PriorContrastiveEstimation,
    _extract_y_and_xi,
)


def _build_design_matrix(xi_list, interaction_pairs):
    X_main = torch.stack(xi_list, dim=1).float()
    if interaction_pairs is not None and len(interaction_pairs) > 0:
        i_idx = torch.tensor(
            [i for i, _ in interaction_pairs], device=X_main.device
        )
        j_idx = torch.tensor(
            [j for _, j in interaction_pairs], device=X_main.device
        )
        X_interact = X_main[..., i_idx] * X_main[..., j_idx]
        return torch.cat([X_main, X_interact], dim=-1)
    return X_main


def _log_py_given_theta_gaussian(
    y, xi_list, z, beta_main, beta_interact, sigma2, interaction_pairs=None,
):
    X_full = _build_design_matrix(xi_list, interaction_pairs)
    T = y.shape[1]

    beta_main_gated = beta_main * z
    if interaction_pairs is not None and len(interaction_pairs) > 0:
        i_idx = torch.tensor(
            [i for i, _ in interaction_pairs], device=z.device
        )
        j_idx = torch.tensor(
            [j for _, j in interaction_pairs], device=z.device
        )
        z_ij = z[..., i_idx] * z[..., j_idx]
        beta_interact_gated = beta_interact * z_ij
        theta_full = torch.cat([beta_main_gated, beta_interact_gated], dim=-1)
    else:
        theta_full = beta_main_gated

    X_exp = X_full.unsqueeze(1)
    theta_exp = theta_full.unsqueeze(2)
    mu = (X_exp * theta_exp).sum(-1)

    y_exp = y.unsqueeze(1)
    resid = y_exp - mu
    sigma2_exp = sigma2.unsqueeze(-1)
    log_p = -0.5 * (
        T * torch.log(2 * math.pi * sigma2_exp.squeeze(-1))
        + (resid.pow(2).sum(-1) / sigma2)
    )
    return log_p


class PriorContrastiveEstimationTotalMI(PriorContrastiveEstimation):

    def _sample_joint_prior(self, primary_trace, num_contrast):
        num_outer, p = primary_trace.nodes["z"]["value"].shape[:2] \
            if primary_trace.nodes["z"]["value"].dim() >= 2 \
            else (primary_trace.nodes["z"]["value"].shape[0],
                  primary_trace.nodes["z"]["value"].shape[-1])
        device = primary_trace.nodes["z"]["value"].device

        z_fn = primary_trace.nodes["z"]["fn"]
        pi = (z_fn.base_dist.probs
              if hasattr(z_fn, "base_dist")
              else z_fn.probs)
        z_con = torch.bernoulli(
            pi.unsqueeze(1).expand(-1, num_contrast, -1)
        )

        scale_main = torch.where(
            z_con.bool(),
            torch.tensor(self.scale_active, device=device),
            torch.tensor(self.scale_inactive, device=device),
        )
        beta_main_con = torch.randn_like(scale_main) * scale_main

        if self.interaction_pairs is not None and len(self.interaction_pairs) > 0:
            i_idx = torch.tensor(
                [i for i, _ in self.interaction_pairs], device=device
            )
            j_idx = torch.tensor(
                [j for _, j in self.interaction_pairs], device=device
            )
            z_ij_con = z_con[..., i_idx] * z_con[..., j_idx]
            scale_int = torch.where(
                z_ij_con.bool(),
                torch.tensor(self.scale_active, device=device),
                torch.tensor(self.scale_inactive, device=device),
            )
            beta_interact_con = torch.randn_like(scale_int) * scale_int
        else:
            beta_interact_con = None

        gamma_sample = torch.distributions.Gamma(
            torch.tensor(self.sigma2_alpha0, device=device),
            torch.tensor(self.sigma2_beta0, device=device),
        ).sample((num_outer, num_contrast))
        sigma2_con = 1.0 / gamma_sample.clamp(min=1e-8)

        return z_con, beta_main_con, beta_interact_con, sigma2_con

    def _gaussian_log_prob(self, primary_trace, num_contrast, p, T):
        y, xi_list = _extract_y_and_xi(primary_trace, self.num_outer_samples, T)

        z_pos = primary_trace.nodes["z"]["value"].float().unsqueeze(1)
        beta_main_pos = primary_trace.nodes["theta_main"]["value"].float().unsqueeze(1)
        if "theta_interact" in primary_trace.nodes:
            beta_interact_pos = primary_trace.nodes["theta_interact"]["value"].float().unsqueeze(1)
        else:
            beta_interact_pos = None
        sigma2_inv_pos = primary_trace.nodes["sigma2_inv"]["value"].float()
        sigma2_pos = (1.0 / sigma2_inv_pos.clamp(min=1e-8)).unsqueeze(1)

        log_py_pos = _log_py_given_theta_gaussian(
            y, xi_list, z_pos, beta_main_pos, beta_interact_pos, sigma2_pos,
            interaction_pairs=self.interaction_pairs,
        ).squeeze(1)

        z_con, beta_main_con, beta_interact_con, sigma2_con = \
            self._sample_joint_prior(primary_trace, num_contrast)
        log_py_con = _log_py_given_theta_gaussian(
            y, xi_list, z_con, beta_main_con, beta_interact_con, sigma2_con,
            interaction_pairs=self.interaction_pairs,
        )

        return log_py_pos, log_py_con

    def differentiable_loss(self, model, *args, **kwargs):
        design_net = None
        for module in (model.__self__.modules()
                       if hasattr(model, '__self__') else []):
            if hasattr(module, '_logits_history'):
                module._logits_history = []
                design_net = module
                break
        if design_net is None:
            try:
                m = model.__self__ if hasattr(model, '__self__') else model
                if hasattr(m, 'design_net') and hasattr(m.design_net, '_logits_history'):
                    m.design_net._logits_history = []
                    design_net = m.design_net
            except Exception:
                pass

        primary_trace = self.get_primary_rollout(model, args, kwargs)
        T = sum(1 for k in primary_trace.nodes if k.startswith("y"))
        p = primary_trace.nodes["z"]["value"].shape[-1]

        log_py_pos, log_py_con = self._gaussian_log_prob(
            primary_trace, self.num_inner_samples, p, T
        )
        log_py_pos_exp = log_py_pos.unsqueeze(1)
        log_py_combined = torch.cat(
            [log_py_pos_exp, log_py_con], dim=1
        ).logsumexp(1)
        loss = (log_py_combined - log_py_pos).mean(0)

        primary_trace = prune_subsample_sites(primary_trace)
        design_sites = sorted(
            [n for n in primary_trace.nodes if n.startswith("xi")],
            key=lambda s: int(s[2:]) if s[2:].isdigit() else 0,
        )
        if len(design_sites) > 1 and self.diversity_penalty > 0.0:
            designs = torch.stack([
                primary_trace.nodes[n]["value"].float() for n in design_sites
            ])
            T_d = designs.shape[0]
            flat = designs.reshape(T_d, -1)
            norms = torch.norm(flat, dim=1, keepdim=True) + 1e-8
            sim = torch.matmul(flat, flat.t()) / (norms * norms.t())
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            loss = loss + self.diversity_penalty * sim[mask].mean()

        if self.logit_l2_penalty > 0.0 and design_net is not None:
            logits_list = getattr(design_net, '_logits_history', [])
            if logits_list:
                all_logits = torch.cat([l.reshape(-1) for l in logits_list])
                loss = loss + self.logit_l2_penalty * all_logits.pow(2).mean()

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, *args, **kwargs))