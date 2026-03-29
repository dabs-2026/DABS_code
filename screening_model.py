import torch
from torch import nn

import pyro
import pyro.distributions as dist

from oed.primitives import observation_sample, latent_sample, compute_design


class ScreeningExperiment(nn.Module):
    def __init__(
        self,
        design_net,
        num_factors=20,
        T=10,
        expected_active_proportion=0.2,
        pi_beta_a=0.0,
        pi_beta_b=0.0,
        scale_active=1.0,
        scale_inactive=0.01,
        noise_scale=0.5,
    ):
        super().__init__()
        self.design_net = design_net
        self.num_factors = num_factors
        self.T = T
        self.pi_fixed = expected_active_proportion
        self.pi_beta_a = pi_beta_a
        self.pi_beta_b = pi_beta_b
        self.use_pi_hyperprior = (pi_beta_a > 0.0 and pi_beta_b > 0.0)
        self.scale_active = scale_active
        self.scale_inactive = scale_inactive
        self.register_buffer('noise_scale', torch.tensor(noise_scale))

        self.interaction_pairs = [
            (i, j)
            for i in range(num_factors)
            for j in range(i + 1, num_factors)
        ]
        self.num_interactions = len(self.interaction_pairs)

    def _sample_pi(self):
        if self.use_pi_hyperprior:
            pi = latent_sample(
                "pi",
                dist.Beta(
                    torch.tensor(self.pi_beta_a, device=self.noise_scale.device),
                    torch.tensor(self.pi_beta_b, device=self.noise_scale.device),
                )
            )
        else:
            pi = self.pi_fixed
        return pi

    def _sample_theta(self, pi):
        device = self.noise_scale.device
        pi_vec = (
            pi.unsqueeze(-1).expand(*pi.shape, self.num_factors)
            if isinstance(pi, torch.Tensor) and pi.dim() > 0
            else pi * torch.ones(self.num_factors, device=device)
        )

        z = latent_sample(
            "z",
            dist.Bernoulli(pi_vec).to_event(1),
            infer={"subtype": "target"}
        )

        scale_main = torch.where(
            z.bool(),
            torch.tensor(self.scale_active, device=device),
            torch.tensor(self.scale_inactive, device=device)
        )
        theta_main = latent_sample(
            "theta_main",
            dist.Normal(
                torch.zeros(self.num_factors, device=device),
                scale_main
            ).to_event(1)
        )

        interact_active = torch.stack([
            z[..., i] * z[..., j]
            for i, j in self.interaction_pairs
        ], dim=-1)

        scale_interact = torch.where(
            interact_active.bool(),
            torch.tensor(self.scale_active, device=device),
            torch.tensor(self.scale_inactive, device=device)
        )
        theta_interact = latent_sample(
            "theta_interact",
            dist.Normal(
                torch.zeros(self.num_interactions, device=device),
                scale_interact
            ).to_event(1)
        )
        return z, theta_main, theta_interact

    def _forward_map(self, xi, theta_main, theta_interact):
        xi = xi.to(theta_main.device)

        while xi.dim() < theta_main.dim():
            xi = xi.unsqueeze(-2)

        mean_y = (xi * theta_main).sum(-1)

        if self.num_interactions > 0:
            xi_interact = torch.stack([
                xi[..., i] * xi[..., j]
                for i, j in self.interaction_pairs
            ], dim=-1)
            mean_y = mean_y + (xi_interact * theta_interact).sum(-1)

        return mean_y

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        pi = self._sample_pi()
        z, theta_main, theta_interact = self._sample_theta(pi)

        y_outcomes = []
        xi_designs = []

        for t in range(self.T):
            xi = compute_design(
                f"xi{t + 1}",
                self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            mean_y = self._forward_map(xi, theta_main, theta_interact)
            y = observation_sample(
                f"y{t + 1}",
                dist.Normal(mean_y, self.noise_scale)
            )
            y_outcomes.append(y)
            xi_designs.append(xi.detach())

        return y_outcomes

    def forward(self, theta=None):
        self.design_net.eval()
        if theta is not None:
            model = pyro.condition(self.model, data=theta)
        else:
            model = self.model

        designs = []
        observations = []

        with torch.no_grad():
            trace = pyro.poutine.trace(model).get_trace()
            for t in range(self.T):
                xi = trace.nodes[f"xi{t + 1}"]["value"]
                designs.append(xi)
                y = trace.nodes[f"y{t + 1}"]["value"]
                observations.append(y)

        designs = [d.squeeze() for d in designs]
        observations = [o.squeeze() for o in observations]
        return torch.stack(designs), torch.stack(observations)

    def eval_policy(self, n_trace=3, theta=None, verbose=True):
        self.design_net.eval()
        if theta is not None:
            model = pyro.condition(self.model, data=theta)
        else:
            model = self.model

        results = []

        with torch.no_grad():
            for i in range(n_trace):
                if verbose:
                    print(f"\nExample run {i + 1}")

                trace = pyro.poutine.trace(model).get_trace()

                true_z = trace.nodes["z"]["value"].cpu()
                true_theta_main = trace.nodes["theta_main"]["value"].cpu()
                active_factors = true_z.nonzero(as_tuple=True)[0].tolist()

                if verbose:
                    if self.use_pi_hyperprior:
                        pi_val = trace.nodes["pi"]["value"].cpu().item()
                        print(f"  Sampled pi:     {pi_val:.3f}")
                    print(f"  Active factors: {active_factors}")
                    print(f"  theta_main:     {true_theta_main.round(decimals=2)}")

                run_designs = []
                run_observations = []

                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].cpu()
                    y = trace.nodes[f"y{t + 1}"]["value"].cpu().item()
                    run_designs.append(xi)
                    run_observations.append(y)

                    if verbose:
                        print(f"  xi{t + 1}: {xi.int().tolist()}  y{t + 1}: {y:.3f}")

                results.append({
                    "z": true_z,
                    "theta_main": true_theta_main,
                    "active_factors": active_factors,
                    "designs": torch.stack(run_designs),
                    "observations": run_observations,
                })

        return results