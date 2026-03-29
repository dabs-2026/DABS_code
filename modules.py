import torch
from torch import nn
import pyro


def rmv(A, b):
    return torch.matmul(A, b.unsqueeze(-1)).squeeze(-1)

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
        batch_shape = xi.shape[:-1]
        p = xi.shape[-1]
        xi_flat = xi.reshape(-1, p)
        if isinstance(y, torch.Tensor):
            y_flat = y.reshape(-1, 1)
        else:
            y_flat = torch.tensor(
                [[y]], dtype=xi.dtype, device=xi.device
            ).expand(xi_flat.shape[0], 1)
        inp = torch.cat([xi_flat, y_flat], dim=-1)
        out = self.network(inp)
        return out.reshape(*batch_shape, self.encoding_dim)

class SufficientStatEncoder(nn.Module):
    def __init__(self, num_factors, encoding_dim=32, proj_hidden=64):
        super().__init__()
        p = num_factors
        self.num_factors  = p
        self.encoding_dim = encoding_dim

        idx = torch.triu_indices(p, p)
        self.register_buffer("triu_row", idx[0])
        self.register_buffer("triu_col", idx[1])

        stat_dim = p * (p + 1) // 2 + p + 1

        self.projection = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, proj_hidden),
            nn.Softplus(),
            nn.Linear(proj_hidden, encoding_dim),
        )

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            return None

        xi_list, y_list = zip(*design_obs_pairs)

        xi = torch.stack(xi_list, dim=-2).float()
        y_raw = []
        for y_s in y_list:
            if isinstance(y_s, torch.Tensor):
                y_raw.append(y_s.float().squeeze(-1))
            else:
                y_raw.append(
                    torch.full(xi.shape[:-2], float(y_s),
                               dtype=xi.dtype, device=xi.device)
                )
        y = torch.stack(y_raw, dim=-1)

        XtX     = torch.matmul(xi.transpose(-1, -2), xi)
        Xty     = torch.matmul(
            xi.transpose(-1, -2), y.unsqueeze(-1)
        ).squeeze(-1)
        yty     = (y * y).sum(-1, keepdim=True)
        XtX_tri = XtX[..., self.triu_row, self.triu_col]

        stats = torch.cat([XtX_tri, Xty, yty], dim=-1)

        return self.projection(stats)

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
            nn.Linear(hidden_dim, num_factors),
        )

    def forward(self, encoding):
        return self.network(encoding)

class SetEquivariantDesignNetwork(nn.Module):
    def __init__(self, encoder_network, emission_network, empty_value,
                 encoder_type="original"):
        super().__init__()
        self.encoder      = encoder_network
        self.emitter      = emission_network
        self.encoder_type = encoder_type
        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))

    def forward(self, *design_obs_pairs):
        t = len(design_obs_pairs)

        if self.encoder_type == "original":
            if t == 0:
                encoding = self.empty_value.new_zeros(
                    (self.encoder.encoding_dim,)
                )
            else:
                encodings = [
                    self.encoder(xi=design, y=obs, t=[idx + 1])
                    for idx, (design, obs) in enumerate(design_obs_pairs)
                ]
                encoding = torch.stack(encodings, dim=0).sum(dim=0)

        else:
            if t == 0:
                encoding = self.encoder.projection[-1].weight.new_zeros(
                    self.encoder.encoding_dim
                )
            else:
                encoding = self.encoder(*design_obs_pairs)

        return self.emitter(encoding)

    def lazy(self, *design_obs_pairs):
        logits = self.forward(*design_obs_pairs)
        logits_2class = torch.stack([-logits, logits], dim=-1)
        soft = torch.nn.functional.gumbel_softmax(
            logits_2class, tau=0.1, hard=True, dim=-1
        )
        xi = soft[..., 1] * 2 - 1
        return pyro.distributions.Delta(xi, event_dim=1)

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
        xi_discrete   = torch.sign(xi_continuous)
        return xi_continuous + (xi_discrete - xi_continuous).detach()


class RandomDesignBaseline(SetEquivariantDesignNetwork):
    def __init__(self, design_dim):
        nn.Module.__init__(self)
        self.register_buffer("prototype", torch.zeros(design_dim))
        self.design_dim = design_dim

    def forward(self, *design_obs_pairs):
        return torch.sign(torch.randn(self.design_dim))

def build_screening_network(
    num_factors, encoding_dim=32, hidden_dim=64,
    encoder_type="original", proj_hidden=64,
):
    if encoder_type == "original":
        encoder      = ScreeningEncoder(
            num_factors=num_factors,
            encoding_dim=encoding_dim,
            hidden_dim=hidden_dim,
        )
        emitter_input = encoding_dim

    else:
        encoder      = SufficientStatEncoder(
            num_factors=num_factors,
            encoding_dim=encoding_dim,
            proj_hidden=proj_hidden,
        )
        emitter_input = encoding_dim

    emitter = ScreeningEmitter(
        encoding_dim=emitter_input,
        num_factors=num_factors,
        hidden_dim=hidden_dim,
    )
    empty_value = torch.zeros(num_factors)

    return SetEquivariantDesignNetwork(
        encoder_network=encoder,
        emission_network=emitter,
        empty_value=empty_value,
        encoder_type=encoder_type,
    )