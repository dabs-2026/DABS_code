import argparse
import torch
import pyro
import pyro.distributions as dist
import math

from modules import build_screening_network
from screening_model import ScreeningExperiment


def run_policy(screening_model, true_z, true_theta_main, true_theta_interact,
               true_sigma2=None, true_z_interact=None):
    true_theta = {
        "z": true_z,
        "theta_main": true_theta_main,
        "theta_interact": true_theta_interact,
    }
    if true_sigma2 is not None:
        true_theta["sigma2_inv"] = 1.0 / torch.as_tensor(true_sigma2).clamp(min=1e-8)
    if true_z_interact is not None:
        true_theta["z_interact"] = true_z_interact
    designs, observations = screening_model.forward(theta=true_theta)
    return designs, observations


def build_design_matrix(designs, interaction_pairs):
    X_main = designs
    if len(interaction_pairs) > 0:
        X_interact = torch.stack([
            designs[:, i] * designs[:, j]
            for i, j in interaction_pairs
        ], dim=-1)
        X = torch.cat([X_main, X_interact], dim=-1)
    else:
        X = X_main
    return X


def _interaction_active_vector(z, z_interact, interaction_pairs):
    if z_interact is not None:
        return z_interact
    return torch.stack([z[i] * z[j] for i, j in interaction_pairs], dim=-1)


def _student_log_marginal(y, X, z, interaction_pairs,
                          scale_active, scale_inactive,
                          alpha0, beta0, z_interact=None):
    T, n_params = X.shape

    scale_main = torch.where(
        z.bool(),
        torch.tensor(scale_active),
        torch.tensor(scale_inactive)
    )
    if len(interaction_pairs) > 0:
        interact_active = _interaction_active_vector(z, z_interact, interaction_pairs)
        scale_interact = torch.where(
            interact_active.bool(),
            torch.tensor(scale_active),
            torch.tensor(scale_inactive)
        )
        prior_std = torch.cat([scale_main, scale_interact])
    else:
        prior_std = scale_main

    prior_var = prior_std ** 2
    K = (X * prior_var.unsqueeze(0)) @ X.T + torch.eye(T)
    Sigma = (beta0 / alpha0) * K

    nu = 2.0 * alpha0
    d  = float(T)

    jitter = 1e-6 * torch.eye(T)
    try:
        L = torch.linalg.cholesky(Sigma + jitter)
    except Exception:
        L = torch.linalg.cholesky(Sigma + 1e-4 * torch.eye(T))

    log_det = 2.0 * L.diagonal().log().sum()

    alpha_vec = torch.linalg.solve_triangular(
        L, y.unsqueeze(-1), upper=False
    ).squeeze(-1)
    quad = (alpha_vec * alpha_vec).sum()

    log_norm = (
        torch.lgamma(torch.tensor((nu + d) / 2.0))
        - torch.lgamma(torch.tensor(nu / 2.0))
        - (d / 2.0) * math.log(nu * math.pi)
    )
    log_p = log_norm - 0.5 * log_det - ((nu + d) / 2.0) * torch.log1p(quad / nu)
    return log_p


def gibbs_sampler(
    designs,
    observations,
    screening_model,
    num_samples=2000,
    burnin=500,
):
    use_pi_hyperprior = screening_model.use_pi_hyperprior
    pi_beta_a         = screening_model.pi_beta_a
    pi_beta_b         = screening_model.pi_beta_b
    p                 = screening_model.num_factors
    interaction_pairs = screening_model.interaction_pairs
    num_interactions  = len(interaction_pairs)
    pi                = screening_model.pi_fixed
    scale_active      = screening_model.scale_active
    scale_inactive    = screening_model.scale_inactive

    alpha0 = screening_model.sigma2_alpha0
    beta0  = screening_model.sigma2_beta0
    use_g_prior = getattr(screening_model, "use_g_prior", False)

    strong_heredity = getattr(screening_model, "strong_heredity", False)
    pi_interaction  = getattr(screening_model, "pi_interaction", 0.5)

    T          = len(designs)
    X          = build_design_matrix(designs, interaction_pairs)
    y          = observations
    num_params = p + num_interactions

    def _prior_std(z, z_int):
        scale_main = torch.where(
            z.bool(),
            torch.tensor(scale_active),
            torch.tensor(scale_inactive),
        )
        if num_interactions > 0:
            interact_active = _interaction_active_vector(z, z_int, interaction_pairs)
            scale_interact = torch.where(
                interact_active.bool(),
                torch.tensor(scale_active),
                torch.tensor(scale_inactive),
            )
            return torch.cat([scale_main, scale_interact])
        return scale_main

    def sample_theta_given_z_sigma2(z, z_int, sigma2):
        prior_std  = _prior_std(z, z_int)
        prior_var  = prior_std ** 2
        prior_prec = torch.diag(1.0 / prior_var)

        XtX = X.T @ X
        if use_g_prior:
            post_prec = (XtX + prior_prec) / sigma2
        else:
            post_prec = XtX / sigma2 + prior_prec
        try:
            L        = torch.linalg.cholesky(post_prec)
            post_cov = torch.cholesky_inverse(L)
        except Exception:
            post_prec = post_prec + 1e-6 * torch.eye(num_params)
            L         = torch.linalg.cholesky(post_prec)
            post_cov  = torch.cholesky_inverse(L)

        post_mean = post_cov @ X.T @ y / sigma2
        theta = dist.MultivariateNormal(post_mean, post_cov).sample()
        return theta

    def sample_sigma2_given_z_theta(z, z_int, theta):
        residual = y - X @ theta
        rss      = (residual * residual).sum()

        if use_g_prior:
            prior_std  = _prior_std(z, z_int)
            prior_var  = prior_std ** 2
            prior_quad = (theta * theta / prior_var).sum()
            d = num_params
            alpha_n = alpha0 + (T + d) / 2.0
            beta_n  = beta0 + 0.5 * rss + 0.5 * prior_quad
        else:
            alpha_n = alpha0 + T / 2.0
            beta_n  = beta0 + 0.5 * rss

        gamma_sample = dist.Gamma(
            torch.tensor(alpha_n),
            torch.tensor(beta_n)
        ).sample()
        sigma2 = 1.0 / gamma_sample.clamp(min=1e-8)
        return sigma2

    pairs_touching = [[] for _ in range(p)]
    for idx, (i, j) in enumerate(interaction_pairs):
        pairs_touching[i].append(idx)
        pairs_touching[j].append(idx)

    def sample_zk_given_rest(z, z_int, k, pi_val):
        z1 = z.clone(); z1[k] = 1.0
        z0 = z.clone(); z0[k] = 0.0
        z_int1 = z_int
        z_int0 = z_int
        if strong_heredity and z_int is not None and num_interactions > 0:
            mask_off = torch.zeros_like(z_int)
            for idx in pairs_touching[k]:
                mask_off[idx] = 1.0
            z_int0 = z_int * (1.0 - mask_off)

        log_p1 = (_student_log_marginal(y, X, z1, interaction_pairs,
                                         scale_active, scale_inactive,
                                         alpha0, beta0, z_interact=z_int1)
                  + math.log(pi_val))
        log_p0 = (_student_log_marginal(y, X, z0, interaction_pairs,
                                         scale_active, scale_inactive,
                                         alpha0, beta0, z_interact=z_int0)
                  + math.log(1.0 - pi_val + 1e-10))

        log_total   = torch.logaddexp(log_p1, log_p0)
        prob_active = torch.exp(log_p1 - log_total).clamp(0.0, 1.0)
        return torch.bernoulli(prob_active)

    def sample_zint_idx_given_rest(z, z_int, idx):
        i, j = interaction_pairs[idx]
        if z[i] < 0.5 or z[j] < 0.5:
            return torch.tensor(0.0)
        z_int1 = z_int.clone(); z_int1[idx] = 1.0
        z_int0 = z_int.clone(); z_int0[idx] = 0.0
        log_p1 = (_student_log_marginal(y, X, z, interaction_pairs,
                                         scale_active, scale_inactive,
                                         alpha0, beta0, z_interact=z_int1)
                  + math.log(pi_interaction + 1e-12))
        log_p0 = (_student_log_marginal(y, X, z, interaction_pairs,
                                         scale_active, scale_inactive,
                                         alpha0, beta0, z_interact=z_int0)
                  + math.log(1.0 - pi_interaction + 1e-12))
        log_total = torch.logaddexp(log_p1, log_p0)
        prob = torch.exp(log_p1 - log_total).clamp(0.0, 1.0)
        return torch.bernoulli(prob)

    sigma2_init = beta0 / (alpha0 - 1.0) if alpha0 > 1.0 else beta0 / alpha0
    sigma2 = torch.tensor(sigma2_init)

    z = dist.Bernoulli(pi * torch.ones(p)).sample()

    if strong_heredity and num_interactions > 0:
        ii = torch.tensor([i for i, _ in interaction_pairs])
        jj = torch.tensor([j for _, j in interaction_pairs])
        parent = z[ii] * z[jj]
        z_int = torch.bernoulli(pi_interaction * parent)
    else:
        z_int = None

    z_samples      = []
    theta_samples  = []
    sigma2_samples = []
    z_int_samples  = []

    total_steps = burnin + num_samples
    print(f"\n  Running Gibbs sampler ({burnin} burnin + {num_samples} samples)...")
    print(f"  NIG prior: alpha0={alpha0:.2f}, beta0={beta0:.4f}  "
          f"(prior mean sigma={sigma2_init**0.5:.4f})")
    if strong_heredity:
        print(f"  Strong heredity ON (Bernoulli z_int, pi_interaction={pi_interaction:.2f})")

    for step in range(total_steps):
        if step % 200 == 0:
            active = z.sum().int().item()
            print(f"    Step {step:>4}/{total_steps}  "
                  f"active={active}  sigma={sigma2.item()**0.5:.4f}")

        theta  = sample_theta_given_z_sigma2(z, z_int, sigma2)
        sigma2 = sample_sigma2_given_z_theta(z, z_int, theta)

        for k in range(p):
            z[k] = sample_zk_given_rest(z, z_int, k, pi if not use_pi_hyperprior else pi)

        if strong_heredity and num_interactions > 0:
            parent_now = z[ii] * z[jj]
            z_int = z_int * parent_now
            for idx in range(num_interactions):
                i, j = interaction_pairs[idx]
                if z[i] >= 0.5 and z[j] >= 0.5:
                    z_int[idx] = sample_zint_idx_given_rest(z, z_int, idx)

        if use_pi_hyperprior:
            a_post = pi_beta_a + z.sum().item()
            b_post = pi_beta_b + p - z.sum().item()
            pi = dist.Beta(
                torch.tensor(a_post),
                torch.tensor(b_post)
            ).sample().item()

        if step >= burnin:
            z_samples.append(z.clone())
            theta_samples.append(theta.clone())
            sigma2_samples.append(sigma2.clone())
            if z_int is not None:
                z_int_samples.append(z_int.clone())

    z_samples      = torch.stack(z_samples)
    theta_samples  = torch.stack(theta_samples)
    sigma2_samples = torch.stack(sigma2_samples)
    z_int_samples  = torch.stack(z_int_samples) if z_int_samples else None

    return z_samples, theta_samples, sigma2_samples, z_int_samples

if __name__ == "__main__":
    from run_eval import main as _main
    _main()
