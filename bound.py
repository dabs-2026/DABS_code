import math

import torch
from torch.distributions.utils import broadcast_all

import pyro
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand


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
    def __init__(self, num_outer_samples=100, num_inner_samples=10, data_source=None):
        super().__init__(num_outer_samples, data_source=data_source)
        self.num_inner_samples = num_inner_samples

    def compute_observation_log_prob(self, trace):
        return sum(
            node["log_prob"]
            for node in trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        )

    def get_contrastive_rollout(
        self,
        trace,
        model,
        args,
        kwargs,
        existing_vectorization,
        graph_type="flat",
        detach=False,
    ):
        sampled_observation_values = {
            name: lexpand(node["value"], self.num_inner_samples)
            for name, node in trace.nodes.items()
            if node.get("subtype") in ["observation_sample", "design_sample"]
        }
        conditional_model = self._vectorized(
            pyro.condition(model, data=sampled_observation_values),
            self.num_inner_samples,
            *existing_vectorization,
            name="inner_vectorization",
        )
        trace = poutine.trace(conditional_model, graph_type=graph_type).get_trace(
            *args, **kwargs
        )
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()

        return trace

    def differentiable_loss(self, model, *args, **kwargs):
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, model, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)

        obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_contrastive]
        ).logsumexp(0)

        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        loss_to_constant = torch_item(self.differentiable_loss(model, *args, **kwargs))
        return loss_to_constant - math.log(self.num_inner_samples + 1)

class NestedMonteCarloEstimation(PriorContrastiveEstimation):
    def differentiable_loss(self, model, *args, **kwargs):
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, model, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)
        obs_log_prob_combined = obs_log_prob_contrastive.logsumexp(0)
        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        loss_to_constant = torch_item(self.differentiable_loss(model, *args, **kwargs))
        return loss_to_constant - math.log(self.num_inner_samples)

class PriorContrastiveEstimationScoreGradient(PriorContrastiveEstimation):
    def __init__(self, num_outer_samples, num_inner_samples):
        super().__init__(
            num_outer_samples=num_outer_samples, 
            num_inner_samples=num_inner_samples
        )

    def differentiable_loss(self, model, *args, **kwargs):
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, model, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)
        obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_contrastive]
        ).logsumexp(0)

        with torch.no_grad():
            g_no_grad = obs_log_prob_primary - obs_log_prob_combined

        loss = -(g_no_grad * obs_log_prob_primary - obs_log_prob_combined).mean(0)

        design_sites = [
            name for name in primary_trace.nodes
            if primary_trace.nodes[name].get("subtype") == "design_sample"
        ]
        if len(design_sites) > 1:
            designs = torch.stack([
                primary_trace.nodes[name]["value"].float()
                for name in design_sites
            ], dim=0)
            d1 = designs[:-1]
            d2 = designs[1:]
            similarity = (d1 * d2).sum(-1) / (d1.norm(dim=-1) * d2.norm(dim=-1) + 1e-8)
            diversity_penalty = similarity.mean()
            loss = loss + 0.1 * diversity_penalty

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        basic_pce = super().differentiable_loss(model, *args, **kwargs)
        loss_to_constant = torch_item(basic_pce)
        return loss_to_constant - math.log(self.num_inner_samples + 1)
