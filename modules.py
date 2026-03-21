import torch
from torch import nn
from pyro.distributions import Delta
import pyro

def rmv(A, b):
    return torch.matmul(A, b.unsqueeze(-1)).squeeze(-1)


class LazyDelta(Delta):
    def __init__(self, fn, prototype, log_density=0.0, event_dim=0, validate_args=None):
        self.fn = fn
        super().__init__(
            prototype,
            log_density=log_density,
            event_dim=event_dim,
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LazyDelta, _instance)
        new.fn = self.fn
        batch_shape = torch.Size(batch_shape)
        new.v = self.v.expand(batch_shape + self.event_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        output = self.fn()
        if sample_shape:
            output = output.expand(sample_shape + output.shape)
        return output

    @property
    def variance(self):
        return torch.zeros_like(self.v)

    def log_prob(self, x):
        return self.log_density


class ScreeningEncoder(nn.Module):
    def __init__(self, num_factors, encoding_dim, hidden_dim=64):
        super().__init__()
        self.encoding_dim = encoding_dim
        input_dim = num_factors + 1

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, encoding_dim),
        )

    def forward(self, xi, y, t=None):
        if isinstance(y, torch.Tensor):
            y_val = y.flatten()[0:1]
        else:
            y_val = torch.tensor([y])
        xi_val = xi.reshape(-1)[:self.network[0].in_features - 1]
        pair = torch.cat([xi_val, y_val], dim=0)
        return self.network(pair)

class ScreeningEmitter(nn.Module):
    def __init__(self, encoding_dim, num_factors, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_factors)
        )

    def forward(self, encoding):
        return self.network(encoding)


class SetEquivariantDesignNetwork(nn.Module):
    def __init__(self, encoder_network, emission_network, empty_value):
        super().__init__()
        self.encoder = encoder_network
        self.emitter = emission_network
        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))

    def lazy(self, *design_obs_pairs):
        logits = self.forward(*design_obs_pairs)
        logits_2class = torch.stack([-logits, logits], dim=-1)
        soft = torch.nn.functional.gumbel_softmax(
        logits_2class, tau=0.1, hard=True, dim=-1
    )
        xi = soft[..., 1] * 2 - 1
        return pyro.distributions.Delta(xi, event_dim=1)

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            sum_encoding = self.empty_value.new_zeros(
                (self.encoder.encoding_dim,)
            )
        else:
            encodings = [
                self.encoder(xi=design, y=obs, t=[idx + 1])
                for idx, (design, obs) in enumerate(design_obs_pairs)
            ]
            sum_encoding = torch.stack(encodings, dim=0).sum(dim=0)
        return self.emitter(sum_encoding)

class BatchDesignBaseline(SetEquivariantDesignNetwork):
    def __init__(self, T, design_dim):
        nn.Module.__init__(self)
        self.register_buffer("prototype", torch.zeros(design_dim))
        self.designs = nn.ParameterList(
            [nn.Parameter(torch.randn(design_dim)) for _ in range(T)]
        )

    def forward(self, *design_obs_pairs):
        t = len(design_obs_pairs)
        xi_continuous = self.designs[t]
        xi_discrete = torch.sign(xi_continuous)
        return xi_continuous + (xi_discrete - xi_continuous).detach()


class RandomDesignBaseline(SetEquivariantDesignNetwork):
    def __init__(self, design_dim):
        nn.Module.__init__(self)
        self.register_buffer("prototype", torch.zeros(design_dim))
        self.design_dim = design_dim

    def forward(self, *design_obs_pairs):
        return torch.sign(
            torch.randn(self.design_dim)
        )


def build_screening_network(num_factors, encoding_dim=16, hidden_dim=64):
    encoder = ScreeningEncoder(
        num_factors=num_factors,
        encoding_dim=encoding_dim,
        hidden_dim=hidden_dim
    )
    emitter = ScreeningEmitter(
        encoding_dim=encoding_dim,
        num_factors=num_factors,
        hidden_dim=hidden_dim
    )
    empty_value = torch.zeros(num_factors)

    return SetEquivariantDesignNetwork(
        encoder_network=encoder,
        emission_network=emitter,
        empty_value=empty_value
    )
