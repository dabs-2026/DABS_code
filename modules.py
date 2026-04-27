import torch
from torch import nn
import pyro

def rmv(A, b):
    return torch.matmul(A, b.unsqueeze(-1)).squeeze(-1)

class ScreeningEncoder(nn.Module):
    def __init__(self, num_factors, encoding_dim, hidden_dim=64, pe_dim=4):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.pe_dim = pe_dim
        input_dim = num_factors + 1 + 2 * pe_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.LayerNorm(encoding_dim),
        )

    def _positional(self, t_val, device, dtype):
        freqs = torch.arange(1, self.pe_dim + 1, device=device, dtype=dtype)
        angles = t_val * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, xi, y, t=None):
        batch_shape = xi.shape[:-1]
        p = xi.shape[-1]
        xi_flat = xi.reshape(-1, p)

        if isinstance(y, torch.Tensor):
            y_flat = y.reshape(-1, 1)
        else:
            y_flat = torch.tensor([[y]], dtype=xi.dtype, device=xi.device).expand(xi_flat.shape[0], 1)

        if t is None:
            t_val = 1.0
        elif isinstance(t, (list, tuple)):
            t_val = float(t[0])
        elif isinstance(t, torch.Tensor):
            t_val = float(t.reshape(-1)[0].item())
        else:
            t_val = float(t)

        pe = self._positional(
            torch.tensor(t_val, dtype=xi.dtype, device=xi.device),
            device=xi.device, dtype=xi.dtype,
        )
        pe_flat = pe.unsqueeze(0).expand(xi_flat.shape[0], -1)

        inp = torch.cat([xi_flat, y_flat, pe_flat], dim=-1)
        out = self.network(inp)
        return out.reshape(*batch_shape, self.encoding_dim)

class SufficientStatEncoder(nn.Module):
    def __init__(self, num_factors, encoding_dim=32, proj_hidden=64):
        super().__init__()
        p = num_factors
        self.num_factors = p
        self.encoding_dim = encoding_dim

        main_i, main_j = [], []
        for i in range(p):
            for j in range(i + 1, p):
                main_i.append(i)
                main_j.append(j)

        self.register_buffer("pair_i", torch.tensor(main_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(main_j, dtype=torch.long))

        d_full = p + p * (p - 1) // 2
        self.d_full = d_full

        idx = torch.triu_indices(d_full, d_full)
        self.register_buffer("triu_row", idx[0])
        self.register_buffer("triu_col", idx[1])

        stat_dim = d_full * (d_full + 1) // 2 + d_full + 1

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
        T = xi.shape[-2]

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

        xi_interact = xi[..., self.pair_i] * xi[..., self.pair_j]
        xi_full = torch.cat([xi, xi_interact], dim=-1)

        XtX = torch.matmul(xi_full.transpose(-1, -2), xi_full) / float(T)
        Xty = torch.matmul(
            xi_full.transpose(-1, -2), y.unsqueeze(-1)
        ).squeeze(-1) / float(T)
        yty = (y * y).mean(-1, keepdim=True)

        XtX_tri = XtX[..., self.triu_row, self.triu_col]

        stats = torch.cat([XtX_tri, Xty, yty], dim=-1)
        return self.projection(stats)

class CanonicalEncoder(nn.Module):
    def __init__(self, num_factors, hidden_dim=64, T_max=20):
        super().__init__()
        self.num_factors = num_factors
        self.T_max = T_max
        self.encoding_dim = hidden_dim

        feature_dim = 3 * num_factors + 2
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)

    def _descending_lexicographic_order(self, *keys):
        order = torch.argsort(-keys[-1], dim=-1, stable=True)
        for key in reversed(keys[:-1]):
            key_sorted = torch.gather(key, -1, order)
            key_order = torch.argsort(-key_sorted, dim=-1, stable=True)
            order = torch.gather(order, -1, key_order)
        return order

    def _lexicographic_compare(self, lhs, rhs):
        diff = lhs - rhs
        neq = diff != 0
        any_neq = neq.any(dim=-1)
        first_idx = neq.float().argmax(dim=-1)
        first_diff = torch.gather(diff, -1, first_idx.unsqueeze(-1)).squeeze(-1)
        return torch.where(any_neq, torch.where(first_diff > 0, 1.0, -1.0), 0.0)

    def canonicalize(self, XtX, Xty, t):
        p = self.num_factors
        t_safe = max(t, 1.0)
        xty_norm = Xty / t_safe
        xtx_norm = XtX / t_safe

        xty_neighbors = xty_norm.unsqueeze(-2).expand_as(xtx_norm)
        abs_xty = xty_neighbors.abs()
        abs_xtx = xtx_norm.abs()
        signed_products = xty_neighbors * xtx_norm

        order_plus = self._descending_lexicographic_order(
            abs_xty, abs_xtx, signed_products
        )
        order_minus = self._descending_lexicographic_order(
            abs_xty, abs_xtx, -signed_products
        )

        def serialize(order, signed_vals):
            sorted_abs_xty = torch.gather(abs_xty, -1, order)
            sorted_abs_xtx = torch.gather(abs_xtx, -1, order)
            sorted_signed = torch.gather(signed_vals, -1, order)
            triple = torch.stack([sorted_abs_xty, sorted_abs_xtx, sorted_signed], dim=-1)
            return triple.reshape(*triple.shape[:-2], 3 * p)

        features_plus = serialize(order_plus, signed_products)
        features_minus = serialize(order_minus, -signed_products)

        orientation = self._lexicographic_compare(
            features_plus, features_minus
        ).to(XtX.dtype)

        canonical = torch.where(
            (orientation >= 0).unsqueeze(-1),
            features_plus,
            features_minus,
        )
        return canonical, orientation

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            return None

        xi_list, y_list = zip(*design_obs_pairs)
        xi = torch.stack(xi_list, dim=-2).float()
        T = xi.shape[-2]

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

        XtX = torch.matmul(xi.transpose(-1, -2), xi)
        Xty = torch.matmul(
            xi.transpose(-1, -2), y.unsqueeze(-1)
        ).squeeze(-1)
        yty = (y * y).sum(-1, keepdim=True)

        canonical, orientation = self.canonicalize(XtX, Xty, float(T))

        t_safe = max(float(T), 1.0)
        p = self.num_factors
        yty_feat = (yty / t_safe).unsqueeze(-2).expand(
            *yty.shape[:-1], p, 1
        )
        t_feat = torch.full_like(yty_feat, float(T) / self.T_max)
        features = torch.cat([canonical, yty_feat, t_feat], dim=-1)

        logits_raw = self.mlp(features).squeeze(-1)
        logits = orientation * logits_raw

        return logits

class ScreeningEmitter(nn.Module):
    def __init__(self, encoding_dim, num_factors, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, num_factors),
        )
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, encoding):
        return self.network(encoding)

class SetEquivariantDesignNetwork(nn.Module):
    def __init__(self, encoder_network, emission_network, empty_value,
                 encoder_type="original"):
        super().__init__()
        self.encoder      = encoder_network
        self.emitter      = emission_network
        self.encoder_type = encoder_type
        self.tau = 1.0
        self._has_static_floor = False
        self.register_buffer("prototype", empty_value.clone())

        enc_dim = self.encoder.encoding_dim
        self.register_parameter(
            "empty_value",
            nn.Parameter(torch.randn(enc_dim)),
        )

    def warm_start_from_static(self, static_checkpoint_path, scale=3.0):
        ckpt = torch.load(static_checkpoint_path, map_location="cpu",
                          weights_only=False)
        state = ckpt["model_state_dict"]
        designs = []
        t = 0
        while f"design_net.designs.{t}" in state:
            designs.append(state[f"design_net.designs.{t}"].clone())
            t += 1
        if not designs:
            raise ValueError(f"No static designs found in {static_checkpoint_path}")
        designs = torch.stack(designs, dim=0)
        target = designs.mean(dim=0)
        bias_init = scale * torch.sign(target)
        with torch.no_grad():
            self.emitter.network[-1].weight.mul_(0.01)
            self.emitter.network[-1].bias.copy_(bias_init)
        print(f"Warm-started emitter bias from {static_checkpoint_path} "
              f"(||target||={target.norm():.3f}, scale={scale})")

    def load_static_floor(self, static_checkpoint_path):
        ckpt = torch.load(static_checkpoint_path, map_location="cpu",
                          weights_only=False)
        state = ckpt["model_state_dict"]
        designs = []
        t = 0
        while f"design_net.designs.{t}" in state:
            designs.append(state[f"design_net.designs.{t}"].clone())
            t += 1
        if not designs:
            raise ValueError(f"No static designs found in {static_checkpoint_path}")
        for i, d in enumerate(designs):
            self.register_buffer(f"static_floor_{i}", d)
        self._n_static = len(designs)
        self._has_static_floor = True
        print(f"Loaded static floor: {len(designs)} designs from {static_checkpoint_path}")

    def forward(self, *design_obs_pairs):
        t = len(design_obs_pairs)

        if self.encoder_type == "canonical":
            if t == 0:
                adaptive_logits = torch.zeros(self.encoder.num_factors,
                                              device=self.prototype.device)
            else:
                adaptive_logits = self.encoder(*design_obs_pairs)
            if getattr(self, "_has_static_floor", False) and t < getattr(self, "_n_static", 0):
                static_logits = getattr(self, f"static_floor_{t}").to(adaptive_logits.device)
                return static_logits + adaptive_logits
            return adaptive_logits

        elif self.encoder_type == "original":
            if t == 0:
                encoding = self.empty_value
            else:
                encodings = [
                    self.encoder(xi=design, y=obs, t=[idx + 1])
                    for idx, (design, obs) in enumerate(design_obs_pairs)
                ]
                encoding = torch.stack(encodings, dim=0).sum(dim=0)
        else:
            if t == 0:
                encoding = self.empty_value
            else:
                encoding = self.encoder(*design_obs_pairs)

        adaptive_logits = self.emitter(encoding)

        if getattr(self, "_has_static_floor", False) and t < getattr(self, "_n_static", 0):
            static_logits = getattr(self, f"static_floor_{t}").to(adaptive_logits.device)
            return static_logits + adaptive_logits
        return adaptive_logits

    def lazy(self, *design_obs_pairs):
        t = len(design_obs_pairs)
        logits = self.forward(*design_obs_pairs)

        if not hasattr(self, "_logits_history"):
            self._logits_history = []
        if getattr(self, "_has_static_floor", False) and t < getattr(self, "_n_static", 0):
            static_logits = getattr(self, f"static_floor_{t}").to(logits.device)
            self._logits_history.append(logits - static_logits)
        else:
            self._logits_history.append(logits)

        if self.training:
            logits_2class = torch.stack([-logits, logits], dim=-1)
            soft = torch.nn.functional.gumbel_softmax(
                logits_2class, tau=self.tau, hard=True, dim=-1
            )
            xi = soft[..., 1] * 2 - 1
        else:
            xi = torch.sign(logits)
            xi = torch.where(xi == 0, torch.ones_like(xi), xi)
        return pyro.distributions.Delta(xi, event_dim=1)

class BatchDesignBaseline(SetEquivariantDesignNetwork):
    def __init__(self, T, design_dim):
        nn.Module.__init__(self)
        self.tau = 1.0
        self.register_buffer("prototype", torch.zeros(design_dim))
        self.designs = nn.ParameterList(
            [nn.Parameter(torch.randn(design_dim)) for _ in range(T)]
        )

    def forward(self, *design_obs_pairs):
        t = len(design_obs_pairs)
        xi_continuous = self.designs[t]
        xi_discrete   = torch.sign(xi_continuous)
        return xi_continuous + (xi_discrete - xi_continuous).detach()

    def lazy(self, *design_obs_pairs):
        if not hasattr(self, "_logits_history"):
            self._logits_history = []
        t = len(design_obs_pairs)
        self._logits_history.append(self.designs[t])

        xi = self.forward(*design_obs_pairs)
        return pyro.distributions.Delta(xi, event_dim=1)

class RandomDesignBaseline(SetEquivariantDesignNetwork):
    def __init__(self, design_dim):
        nn.Module.__init__(self)
        self.tau = 1.0
        self.register_buffer("prototype", torch.zeros(design_dim))
        self.design_dim = design_dim

    def forward(self, *design_obs_pairs):
        return torch.sign(torch.randn(self.design_dim))

def build_screening_network(
    num_factors, encoding_dim=32, hidden_dim=64,
    encoder_type="original", proj_hidden=64, T_max=20,
):
    if encoder_type == "canonical":
        encoder = CanonicalEncoder(
            num_factors=num_factors,
            hidden_dim=hidden_dim,
            T_max=T_max,
        )
        emitter = nn.Identity()
        empty_value = torch.zeros(num_factors)
        return SetEquivariantDesignNetwork(
            encoder_network=encoder,
            emission_network=emitter,
            empty_value=empty_value,
            encoder_type=encoder_type,
        )

    if encoder_type == "original":
        encoder = ScreeningEncoder(
            num_factors=num_factors,
            encoding_dim=encoding_dim,
            hidden_dim=hidden_dim,
        )
        emitter_input = encoding_dim
    else:
        encoder = SufficientStatEncoder(
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