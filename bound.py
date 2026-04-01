import math

import torch

import pyro
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand

def _log_py_given_z_student(
    y,
    xi_list,
    z_outer,
    scale_active,
    scale_inactive,
    sigma2_alpha0,
    sigma2_beta0,
    interaction_pairs=None,
):
    num_outer, S, p = z_outer.shape
    T = y.shape[1]

    X_main = torch.stack(xi_list, dim=1).float()

    prior_var_main = torch.where(
        z_outer.bool(),
        torch.tensor(scale_active ** 2,   device=z_outer.device),
        torch.tensor(scale_inactive ** 2, device=z_outer.device),
    )

    if interaction_pairs is not None and len(interaction_pairs) > 0:
        i_idx = torch.tensor([i for i, j in interaction_pairs], device=z_outer.device)
        j_idx = torch.tensor([j for i, j in interaction_pairs], device=z_outer.device)
        X_interact = X_main[..., i_idx] * X_main[..., j_idx]
        z_i = z_outer[..., i_idx]
        z_j = z_outer[..., j_idx]
        interact_active = z_i * z_j
        prior_var_interact = torch.where(
            interact_active.bool(),
            torch.tensor(scale_active ** 2,   device=z_outer.device),
            torch.tensor(scale_inactive ** 2, device=z_outer.device),
        )

        X_full        = torch.cat([X_main, X_interact], dim=-1)
        prior_var_full = torch.cat([prior_var_main, prior_var_interact], dim=-1)
    else:
        X_full        = X_main
        prior_var_full = prior_var_main

    X_exp = X_full.unsqueeze(1)
    Xv    = X_exp * prior_var_full.unsqueeze(2)
    K     = torch.matmul(Xv, X_exp.transpose(-1, -2))
    eye   = torch.eye(T, device=K.device).unsqueeze(0).unsqueeze(0)
    K     = K + eye
    scale_factor = sigma2_beta0 / sigma2_alpha0
    Sigma = scale_factor * K

    nu = 2.0 * sigma2_alpha0
    d  = float(T)

    Sigma_flat = Sigma.reshape(num_outer * S, T, T)
    jitter = 1e-6 * torch.eye(T, device=Sigma_flat.device).unsqueeze(0)
    try:
        L = torch.linalg.cholesky(Sigma_flat + jitter)
    except Exception:
        L = torch.linalg.cholesky(
            Sigma_flat + 1e-4 * torch.eye(T, device=Sigma_flat.device).unsqueeze(0)
        )

    log_det_Sigma = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    y_exp  = y.unsqueeze(1).expand(num_outer, S, T)
    y_flat = y_exp.reshape(num_outer * S, T)

    alpha = torch.linalg.solve_triangular(
        L, y_flat.unsqueeze(-1), upper=False
    ).squeeze(-1)
    quad = (alpha * alpha).sum(-1)

    log_norm = (
        torch.lgamma(torch.tensor((nu + d) / 2.0))
        - torch.lgamma(torch.tensor(nu / 2.0))
        - (d / 2.0) * math.log(nu * math.pi)
    )
    log_p = (
        log_norm
        - 0.5 * log_det_Sigma
        - ((nu + d) / 2.0) * torch.log1p(quad / nu)
    )

    return log_p.reshape(num_outer, S)


def _extract_y_and_xi(primary_trace, num_outer, T):
    y_list, xi_list = [], []
    for t in range(1, T + 1):
        y_node  = primary_trace.nodes.get(f"y{t}")
        xi_node = primary_trace.nodes.get(f"xi{t}")
        if y_node is None or xi_node is None:
            break
        y_list.append(y_node["value"].float())
        xi_list.append(xi_node["value"].float())

    y = torch.stack(
        [v.squeeze(-1) if v.dim() > 1 else v for v in y_list], dim=-1
    )
    return y, xi_list

class MutualInformation(object):
    def __init__(self, num_outer_samples, data_source=None):
        self.data_source = data_source
        self.num_outer_samples = num_outer_samples

    def _vectorized(self, fn, *shape, name="vectorization_plate"):
        def wrapped_fn(*args, **kwargs):
            with pyro.plate_stack(name, shape):
                return fn(*args, **kwargs)
        return wrapped_fn

    def get_primary_rollout(self, model, args, kwargs, graph_type="flat", detach=False):
        if self.data_source is None:
            model = self._vectorized(
                model, self.num_outer_samples, name="outer_vectorization"
            )
        else:
            data = next(self.data_source)
            model = pyro.condition(
                self._vectorized(
                    model, self.num_outer_samples, name="outer_vectorization"
                ),
                data=data,
            )
        trace = poutine.trace(model, graph_type=graph_type).get_trace(*args, **kwargs)
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()
        return trace

class PriorContrastiveEstimation(MutualInformation):
    def __init__(
        self,
        num_outer_samples=100,
        num_inner_samples=10,
        data_source=None,
        scale_active=1.0,
        scale_inactive=0.01,
        sigma2_alpha0=3.0,
        sigma2_beta0=None,
        noise_scale=0.5,
        diversity_penalty=0.1,
        interaction_pairs=None,
    ):
        super().__init__(num_outer_samples, data_source=data_source)
        self.num_inner_samples = num_inner_samples
        self.scale_active      = scale_active
        self.scale_inactive    = scale_inactive
        self.diversity_penalty = diversity_penalty
        self.interaction_pairs = interaction_pairs

        self.sigma2_alpha0 = sigma2_alpha0
        if sigma2_beta0 is None:
            sigma2_beta0 = (sigma2_alpha0 + 1.0) * (noise_scale ** 2)
        self.sigma2_beta0 = sigma2_beta0

    def _student_log_prob(self, primary_trace, num_z_contrastive, p, T):
        y, xi_list = _extract_y_and_xi(primary_trace, self.num_outer_samples, T)

        z_pos     = primary_trace.nodes["z"]["value"].float()
        z_pos_exp = z_pos.unsqueeze(1)

        log_py_z_pos = _log_py_given_z_student(
            y, xi_list, z_pos_exp,
            self.scale_active, self.scale_inactive,
            self.sigma2_alpha0, self.sigma2_beta0,
            interaction_pairs=self.interaction_pairs,
        ).squeeze(1)
        z_fn = primary_trace.nodes["z"]["fn"]
        pi   = (z_fn.base_dist.probs
                if hasattr(z_fn, "base_dist")
                else z_fn.probs)

        z_con = torch.bernoulli(
            pi.unsqueeze(1).expand(-1, num_z_contrastive, -1)
        )

        log_py_z_con = _log_py_given_z_student(
            y, xi_list, z_con,
            self.scale_active, self.scale_inactive,
            self.sigma2_alpha0, self.sigma2_beta0,
            interaction_pairs=self.interaction_pairs,
        )

        return log_py_z_pos, log_py_z_con


    def differentiable_loss(self, model, *args, **kwargs):
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        T = sum(1 for k in primary_trace.nodes if k.startswith("y"))
        p = primary_trace.nodes["z"]["value"].shape[-1]

        log_py_z_pos, log_py_z_con = self._student_log_prob(
            primary_trace, self.num_inner_samples, p, T
        )

        log_py_z_pos_exp = log_py_z_pos.unsqueeze(1)
        log_py_combined  = torch.cat(
            [log_py_z_pos_exp, log_py_z_con], dim=1
        ).logsumexp(1)

        loss = (log_py_combined - log_py_z_pos).mean(0)

        design_sites = [
            name for name in primary_trace.nodes
            if primary_trace.nodes[name].get("subtype") == "design_sample"
        ]
        if len(design_sites) > 1 and self.diversity_penalty > 0.0:
            designs = torch.stack([
                primary_trace.nodes[name]["value"].float()
                for name in design_sites
            ])
            flat = designs.reshape(designs.shape[0], -1)
            norms = torch.norm(flat, dim=1, keepdim=True) + 1e-8
            sim_matrix = torch.matmul(flat, flat.t()) / (norms * norms.t())

            mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
            avg_sim = sim_matrix[mask].mean()

            loss = loss + self.diversity_penalty * avg_sim
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        loss_val = torch_item(self.differentiable_loss(model, *args, **kwargs))
        return loss_val - math.log(self.num_inner_samples + 1)

class NestedMonteCarloEstimation(PriorContrastiveEstimation):
    def differentiable_loss(self, model, *args, **kwargs):
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        T = sum(1 for k in primary_trace.nodes if k.startswith("y"))
        p = primary_trace.nodes["z"]["value"].shape[-1]

        log_py_z_pos, log_py_z_con = self._student_log_prob(
            primary_trace, self.num_inner_samples, p, T
        )

        log_py_combined = log_py_z_con.logsumexp(1)
        loss = (log_py_combined - log_py_z_pos).mean(0)
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        loss_val = torch_item(self.differentiable_loss(model, *args, **kwargs))
        return loss_val - math.log(self.num_inner_samples)