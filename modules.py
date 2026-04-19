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
        # (xi, y, sin/cos positional features)
        input_dim = num_factors + 1 + 2 * pe_dim

        # Shallow encoder: one hidden layer only. Matches the FFN style of
        # Strouwen et al. 2026 (their transformer uses 32->64->32 FFN with
        # a single hidden expansion). Smaller networks regularise toward
        # simpler adaptive policies and are much easier to optimise when
        # the PCE gradient is noisy.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, encoding_dim),
            # LayerNorm bounds each encoder contribution to a fixed
            # magnitude so the encoder communicates via direction (which
            # must depend on y) rather than via scale (which the optimiser
            # otherwise inflates to ~44 during training).
            nn.LayerNorm(encoding_dim),
        )

    def _positional(self, t_val, device, dtype):
        # Simple sinusoidal positional encoding with pe_dim frequencies.
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
            y_flat = torch.tensor(
                [[y]], dtype=xi.dtype, device=xi.device
            ).expand(xi_flat.shape[0], 1)

        # Build a time feature. t may be None (t=0 case is handled by
        # empty_value upstream), a list/tuple with the current step, or a
        # tensor. Fall back to t=1 if nothing is provided.
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
    """
    Sufficient-statistic encoder for Bayesian linear regression.

    Key differences from the version used in Strouwen et al. 2026:
      (A) Interaction-aware. The raw design matrix xi [T, p] is expanded to
          include all 2-way interaction columns xi_i * xi_j (i<j), giving
          xi_full [T, d] with d = p + p*(p-1)/2. Sufficient statistics are
          computed on this expanded design matrix, so the encoder receives
          information about how the designs probe interaction effects --
          not only main effects. This matters for screening models where
          theta_interact is part of the generative model.
      (B) T-normalized. X^T X, X^T y, and y^T y are divided by T before
          projection. This keeps the statistics dimensionless and on a
          consistent scale across sequence lengths, which fixes a training
          instability we observed with the un-normalized version (a 1-in-3
          divergence rate during Gumbel-ST training).

    Stat dimension grows as O(d^2) = O(p^4) -- for p=10 we get 1596 stats,
    for p=15 we get ~6400, for p=20 we get ~22000. Still cheap up to ~p=20.
    """

    def __init__(self, num_factors, encoding_dim=32, proj_hidden=64):
        super().__init__()
        p = num_factors
        self.num_factors = p
        self.encoding_dim = encoding_dim

        # Indices of 2-way interaction pairs (i < j) on the p main factors
        main_i, main_j = [], []
        for i in range(p):
            for j in range(i + 1, p):
                main_i.append(i)
                main_j.append(j)
        self.register_buffer(
            "pair_i", torch.tensor(main_i, dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.tensor(main_j, dtype=torch.long)
        )

        # Expanded design-matrix dimension
        d_full = p + p * (p - 1) // 2   # = p*(p+1)/2
        self.d_full = d_full

        # Upper-triangular indices of (X_full)^T X_full
        idx = torch.triu_indices(d_full, d_full)
        self.register_buffer("triu_row", idx[0])
        self.register_buffer("triu_col", idx[1])

        # stat_dim = triu(X^T X) + X^T y + y^T y
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

        xi = torch.stack(xi_list, dim=-2).float()   # [..., T, p]
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
        y = torch.stack(y_raw, dim=-1)               # [..., T]

        # Fix A: expand xi with 2-way interaction columns.
        # xi_interact[..., t, k] = xi[..., t, pair_i[k]] * xi[..., t, pair_j[k]]
        xi_interact = xi[..., self.pair_i] * xi[..., self.pair_j]   # [..., T, num_pairs]
        xi_full = torch.cat([xi, xi_interact], dim=-1)               # [..., T, d_full]

        # Fix B: T-normalized sufficient statistics.
        # For +/-1 designs, diag(X^T X / T) == 1 and off-diagonals are in [-1, 1],
        # which makes the LayerNorm front-end well-conditioned.
        XtX = torch.matmul(xi_full.transpose(-1, -2), xi_full) / float(T)       # [..., d, d]
        Xty = torch.matmul(
            xi_full.transpose(-1, -2), y.unsqueeze(-1)
        ).squeeze(-1) / float(T)                                                # [..., d]
        yty = (y * y).mean(-1, keepdim=True)                                    # [..., 1]

        XtX_tri = XtX[..., self.triu_row, self.triu_col]                        # [..., d*(d+1)/2]

        stats = torch.cat([XtX_tri, Xty, yty], dim=-1)                          # [..., stat_dim]
        return self.projection(stats)

class CanonicalEncoder(nn.Module):
    """
    Signed-permutation-equivariant encoder via canonicalization.

    Implements the approach from Strouwen (2026): for each factor i, build a
    per-factor feature vector from its row in X'X and its entry in X'y,
    canonicalize the sign convention, sort neighbors by magnitude, then
    process with a weight-shared MLP.  The result is exactly equivariant to
    any signed permutation of the factors.

    This replaces both the encoder AND the emitter — the forward() method
    returns logits [batch, p] directly.

    Feature vector per factor i (dimension = 3*p + 2):
      For each neighbor j: (|X'y_j|/t, |X'X_{ij}|/t, sign_product)
      sorted in descending lexicographic order, then:
      yty/t, t/T
    """

    def __init__(self, num_factors, hidden_dim=64, T_max=20):
        super().__init__()
        self.num_factors = num_factors
        self.T_max = T_max
        self.encoding_dim = hidden_dim  # needed by SetEquivariantDesignNetwork

        feature_dim = 3 * num_factors + 2  # per-factor feature size
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init final layer bias
        nn.init.zeros_(self.mlp[-1].bias)

    def _descending_lexicographic_order(self, *keys):
        """Stable descending lexicographic sort along the last axis."""
        order = torch.argsort(-keys[-1], dim=-1, stable=True)
        for key in reversed(keys[:-1]):
            key_sorted = torch.gather(key, -1, order)
            key_order = torch.argsort(-key_sorted, dim=-1, stable=True)
            order = torch.gather(order, -1, key_order)
        return order

    def _lexicographic_compare(self, lhs, rhs):
        """Compare flattened sequences lexicographically."""
        diff = lhs - rhs
        neq = diff != 0
        any_neq = neq.any(dim=-1)
        first_idx = neq.float().argmax(dim=-1)
        first_diff = torch.gather(diff, -1, first_idx.unsqueeze(-1)).squeeze(-1)
        return torch.where(any_neq, torch.where(first_diff > 0, 1.0, -1.0), 0.0)

    def canonicalize(self, XtX, Xty, t):
        """
        Build canonical per-factor features and local sign orientations.

        Args:
            XtX: [..., p, p] normalized X'X
            Xty: [..., p]    normalized X'y
            t:   scalar      current step count

        Returns:
            canonical: [..., p, 3*p]  canonical feature block per factor
            orientation: [..., p]     +1 or -1 sign per factor
        """
        p = self.num_factors
        t_safe = max(t, 1.0)
        xty_norm = Xty / t_safe           # [..., p]
        xtx_norm = XtX / t_safe           # [..., p, p]

        # For each factor i, the neighbor features are indexed by j:
        #   |X'y_j|, |X'X_{ij}|, X'y_j * X'X_{ij}
        # xty_neighbors: [..., p, p] — row i has X'y_j for all j
        xty_neighbors = xty_norm.unsqueeze(-2).expand_as(xtx_norm)
        abs_xty = xty_neighbors.abs()
        abs_xtx = xtx_norm.abs()
        signed_products = xty_neighbors * xtx_norm

        # Sort neighbors for + and - orientations
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
            # Stack into [..., p, p, 3] then flatten to [..., p, 3*p]
            triple = torch.stack([sorted_abs_xty, sorted_abs_xtx, sorted_signed], dim=-1)
            return triple.reshape(*triple.shape[:-2], 3 * p)

        features_plus = serialize(order_plus, signed_products)
        features_minus = serialize(order_minus, -signed_products)

        # Pick canonical orientation per factor
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
        """
        Build sufficient stats from history, canonicalize, run MLP.

        Returns: logits [..., p]
        """
        if len(design_obs_pairs) == 0:
            return None

        xi_list, y_list = zip(*design_obs_pairs)
        xi = torch.stack(xi_list, dim=-2).float()   # [..., T, p]
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
        y = torch.stack(y_raw, dim=-1)               # [..., T]

        # Sufficient statistics (main effects only — interactions are
        # determined by main-effect activity via heredity)
        XtX = torch.matmul(xi.transpose(-1, -2), xi)           # [..., p, p]
        Xty = torch.matmul(
            xi.transpose(-1, -2), y.unsqueeze(-1)
        ).squeeze(-1)                                            # [..., p]
        yty = (y * y).sum(-1, keepdim=True)                     # [..., 1]

        # Canonicalize
        canonical, orientation = self.canonicalize(XtX, Xty, float(T))

        # Append global features: yty/t and t/T_max
        t_safe = max(float(T), 1.0)
        p = self.num_factors
        yty_feat = (yty / t_safe).unsqueeze(-2).expand(
            *yty.shape[:-1], p, 1
        )                                                        # [..., p, 1]
        t_feat = torch.full_like(yty_feat, float(T) / self.T_max)  # [..., p, 1]
        features = torch.cat([canonical, yty_feat, t_feat], dim=-1)  # [..., p, 3p+2]

        # Weight-shared MLP: same weights for every factor
        logits_raw = self.mlp(features).squeeze(-1)              # [..., p]

        # Undo sign canonicalization
        logits = orientation * logits_raw

        return logits


class ScreeningEmitter(nn.Module):
    def __init__(self, encoding_dim, num_factors, hidden_dim=64):
        super().__init__()
        # Shallow emitter: one hidden layer only, matching the encoder and
        # the Strouwen et al. 2026 FFN style.
        self.network = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, num_factors),
        )
        # Zero the final-layer bias so emitter(zeros) == 0 at init. This
        # encourages the emitter to actually use the encoding rather than
        # collapsing to a bias-only fixed design. Keep default weight init
        # so the emitter has full expressivity.
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
        self._has_static_floor = False  # set by load_static_floor()
        self.register_buffer("prototype", empty_value.clone())
        # The t=0 "initial context" is a learnable vector in encoding space,
        # NOT in design space. This is what lets the policy distinguish
        # "no history yet" from "history that happens to sum to zero".
        #
        # Initialize to a scale comparable to one LayerNormed encoder output
        # (which has std ~1 per component, so typical norm ~sqrt(dim)).
        # Using a tiny scale here would create a magnitude discontinuity
        # between t=0 and t>=1 that the emitter has to learn to handle.
        enc_dim = self.encoder.encoding_dim
        self.register_parameter(
            "empty_value",
            nn.Parameter(torch.randn(enc_dim)),
        )

    def load_static_floor(self, static_checkpoint_path):
        """Load frozen static designs as a floor. DABS outputs
        static_design[t] + adaptive_correction(history).
        This guarantees DABS >= static by construction."""
        ckpt = torch.load(static_checkpoint_path, map_location="cpu",
                          weights_only=False)
        state = ckpt["model_state_dict"]
        # Extract the T static design vectors
        designs = []
        t = 0
        while f"design_net.designs.{t}" in state:
            designs.append(state[f"design_net.designs.{t}"].clone())
            t += 1
        if not designs:
            raise ValueError(f"No static designs found in {static_checkpoint_path}")
        # Register as frozen buffers (no gradients)
        for i, d in enumerate(designs):
            self.register_buffer(f"static_floor_{i}", d)
        self._n_static = len(designs)
        # Use a flag instead of storing the list as an attribute
        # (PyTorch rejects plain lists as module attributes)
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
            # Add static floor if loaded
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

        # Add static floor if loaded: logits = static + adaptive_correction
        if getattr(self, "_has_static_floor", False) and t < getattr(self, "_n_static", 0):
            static_logits = getattr(self, f"static_floor_{t}").to(adaptive_logits.device)
            return static_logits + adaptive_logits
        return adaptive_logits

    def lazy(self, *design_obs_pairs):
        t = len(design_obs_pairs)
        logits = self.forward(*design_obs_pairs)

        # Store raw logits for the L2 penalty in bound.py.
        # When using the static floor, we only penalize the ADAPTIVE CORRECTION
        # (logits - static_floor), not the total. This preserves the guarantee
        # that DABS >= Static: at correction=0, the policy outputs the static
        # design, and the L2 penalty doesn't push it away.
        if not hasattr(self, "_logits_history"):
            self._logits_history = []
        if getattr(self, "_has_static_floor", False) and t < getattr(self, "_n_static", 0):
            static_logits = getattr(self, f"static_floor_{t}").to(logits.device)
            # Only penalize the deviation from static
            self._logits_history.append(logits - static_logits)
        else:
            self._logits_history.append(logits)

        # Gumbel straight-through estimator for binary +/- 1 designs.
        #   Forward:  hard sample from Gumbel(logits), giving ±1
        #   Backward: gradient of soft gumbel_softmax with temperature tau
        # The Gumbel noise provides stochastic exploration during training,
        # which helps the optimizer escape the "fixed batch" local minimum
        # that tanh-ST gets trapped in. tau is annealed from tau_start to
        # tau_end by training.py.
        #
        # Previously tried alternative (deterministic tanh-ST) kept here as
        # a comment for reference:
        #   soft = torch.tanh(logits / max(self.tau, 1e-3))
        #   hard = torch.where(logits >= 0, +1., -1.)
        #   xi = soft + (hard - soft).detach()
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
        # Canonical encoder is self-contained (no separate emitter needed).
        # It returns logits directly via its weight-shared per-factor MLP.
        encoder = CanonicalEncoder(
            num_factors=num_factors,
            hidden_dim=hidden_dim,
            T_max=T_max,
        )
        # Emitter is an identity pass-through since canonical encoder
        # already outputs logits.
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