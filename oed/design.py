import torch
import pyro
from pyro.infer.util import torch_item


class OED:
    def __init__(self, model, optim, loss, **kwargs):

        self.model = model
        self.optim = optim
        self.loss = loss
        super().__init__(**kwargs)

        if not isinstance(optim, pyro.optim.PyroOptim):
            raise ValueError(
                "Optimizer should be an instance of pyro.optim.PyroOptim class."
            )

    def evaluate_loss(self, *args, **kwargs):
        with torch.no_grad():
            loss = self.loss.loss(self.model, *args, **kwargs)
            if isinstance(loss, tuple):
                return type(loss)(map(torch_item, loss))
            else:
                return torch_item(loss)

    def step(self, clip_grads=True, *args, **kwargs):
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = self.loss.differentiable_loss(self.model, *args, **kwargs)
            loss.backward()

        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )
        if clip_grads:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, norm_type="inf")

        self.optim(params)

        pyro.infer.util.zero_grads(params)

        if isinstance(loss, tuple):
            return type(loss)(map(torch_item, loss))
        else:
            return torch_item(loss)