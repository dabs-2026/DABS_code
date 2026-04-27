import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import pyro
import pyro.distributions as dist

from bound import _log_py_given_z_student


def enumerate_designs(p, device="cpu"):
    if p > 15:
        raise ValueError(f"Enumeration infeasible for p={p}; use random_candidates()")
    n = 2 ** p
    idx = torch.arange(n, device=device).unsqueeze(-1)
    bits = ((idx >> torch.arange(p, device=device)) & 1).float()
    return 2.0 * bits - 1.0


def random_candidates(p, n=2000, device="cpu", generator=None):
    return torch.sign(torch.randn(n, p, device=device, generator=generator) + 1e-8)


def one_step_spce_scores(
    candidates,
    history_xi,
    history_y,
    scale_active,
    scale_inactive,
    sigma2_alpha0,
    sigma2_beta0,
    pi_prior,
    interaction_pairs,
    n_contrastive=100,
    n_y_samples=3,
    device="cpu",
    rng_seed=None,
):
    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    C, p = candidates.shape
    t_prev = len(history_xi)

    L = n_contrastive
    S = n_y_samples

    z_prior = torch.bernoulli(pi_prior * torch.ones(L, p, device=device))

    eig_per_candidate = torch.zeros(C, device=device)

    for c in range(C):
        xi_new = candidates[c]
        xi_full_list = history_xi + [xi_new]

        z_outer = torch.bernoulli(pi_prior * torch.ones(S, p, device=device))

        gamma_s = dist.Gamma(
            torch.tensor(sigma2_alpha0), torch.tensor(sigma2_beta0)
        ).sample((S,))
        sigma2_s = 1.0 / gamma_s.clamp(min=1e-8)

        scale_main = torch.where(
            z_outer.bool(),
            torch.tensor(scale_active, device=device),
            torch.tensor(scale_inactive, device=device),
        )
        beta_main = torch.randn(S, p, device=device) * scale_main

        if interaction_pairs and len(interaction_pairs) > 0:
            ni = len(interaction_pairs)
            i_idx = torch.tensor([i for i, j in interaction_pairs], device=device)
            j_idx = torch.tensor([j for i, j in interaction_pairs], device=device)
            interact_active = z_outer[:, i_idx] * z_outer[:, j_idx]
            scale_interact = torch.where(
                interact_active.bool(),
                torch.tensor(scale_active, device=device),
                torch.tensor(scale_inactive, device=device),
            )
            beta_interact = torch.randn(S, ni, device=device) * scale_interact
        else:
            beta_interact = None

        T_new = t_prev + 1
        y_sim = torch.zeros(S, T_new, device=device)

        for t_idx, xi_t in enumerate(xi_full_list):
            mean_main = (xi_t.unsqueeze(0) * beta_main).sum(-1)
            mean_interact = torch.zeros(S, device=device)
            if beta_interact is not None:
                xi_inter = xi_t[i_idx] * xi_t[j_idx]
                mean_interact = (xi_inter.unsqueeze(0) * beta_interact).sum(-1)
            mean_t = mean_main + mean_interact
            noise_t = torch.randn(S, device=device) * sigma2_s.sqrt()
            if t_idx < t_prev:
                y_sim[:, t_idx] = history_y[t_idx]
            else:
                y_sim[:, t_idx] = mean_t + noise_t

        log_pos_list = []
        log_con_list = []
        for s in range(S):
            z_batch = torch.cat([z_outer[s].unsqueeze(0), z_prior], dim=0).unsqueeze(0)
            y_batch = y_sim[s].unsqueeze(0)
            xi_tensor_list = [xi_t.unsqueeze(0) for xi_t in xi_full_list]

            log_p = _log_py_given_z_student(
                y_batch,
                xi_tensor_list,
                z_batch,
                scale_active,
                scale_inactive,
                sigma2_alpha0,
                sigma2_beta0,
                interaction_pairs=interaction_pairs,
            ).squeeze(0)

            log_pos_list.append(log_p[0])
            log_con_list.append(log_p)

        log_pos = torch.stack(log_pos_list)
        log_con = torch.stack(log_con_list)

        log_con_mean = torch.logsumexp(log_con, dim=-1) - math.log(log_con.shape[-1])
        eig_s = log_pos - log_con_mean
        eig_per_candidate[c] = eig_s.mean()

    return eig_per_candidate


def simulate_one_response(xi, true_z, true_theta_main, true_theta_interact, true_sigma2,
                          interaction_pairs, device="cpu"):
    mean_y = (xi * true_theta_main).sum()
    if interaction_pairs and len(interaction_pairs) > 0:
        i_idx = torch.tensor([i for i, j in interaction_pairs], device=device)
        j_idx = torch.tensor([j for i, j in interaction_pairs], device=device)
        xi_inter = xi[i_idx] * xi[j_idx]
        mean_y = mean_y + (xi_inter * true_theta_interact).sum()
    noise = torch.randn(1, device=device).item() * math.sqrt(true_sigma2)
    return mean_y.item() + noise


def run_myopic_rollout(
    true_z, true_theta_main, true_theta_interact, true_sigma2,
    T, p,
    scale_active, scale_inactive,
    sigma2_alpha0, sigma2_beta0,
    pi_prior,
    interaction_pairs,
    n_contrastive=100,
    n_y_samples=3,
    candidates=None,
    device="cpu",
    verbose=False,
):
    history_xi = []
    history_y = []

    if candidates is None:
        candidates = enumerate_designs(p, device=device) if p <= 12 \
            else random_candidates(p, n=2000, device=device)

    for t in range(T):
        eigs = one_step_spce_scores(
            candidates, history_xi, history_y,
            scale_active, scale_inactive,
            sigma2_alpha0, sigma2_beta0,
            pi_prior, interaction_pairs,
            n_contrastive=n_contrastive,
            n_y_samples=n_y_samples,
            device=device,
            rng_seed=None,
        )
        best_idx = eigs.argmax().item()
        best_xi = candidates[best_idx].clone()

        y_obs = simulate_one_response(
            best_xi, true_z, true_theta_main, true_theta_interact, true_sigma2,
            interaction_pairs, device=device,
        )

        history_xi.append(best_xi)
        history_y.append(y_obs)

        if verbose:
            print(f"    step {t+1}/{T}: best-eig={eigs.max().item():.3f}")

    X = torch.stack(history_xi, dim=0)
    y = torch.tensor(history_y, device=device)
    return X, y


def run_simple_gibbs(
    X, y, p, interaction_pairs,
    scale_active, scale_inactive,
    alpha0, beta0, pi_prior,
    num_samples=200, burnin=100, threshold=0.5,
):
    T = X.shape[0]
    if interaction_pairs:
        i_idx = torch.tensor([i for i, j in interaction_pairs])
        j_idx = torch.tensor([j for i, j in interaction_pairs])
        X_interact = X[:, i_idx] * X[:, j_idx]
        X_full = torch.cat([X, X_interact], dim=-1)
    else:
        X_full = X
    d = X_full.shape[-1]

    sigma2 = beta0 / (alpha0 - 1.0) if alpha0 > 1.0 else beta0 / alpha0
    sigma2 = torch.tensor(sigma2)
    z = dist.Bernoulli(pi_prior * torch.ones(p)).sample()

    z_samples = []
    total_steps = burnin + num_samples
    for step in range(total_steps):
        scale_main = torch.where(z.bool(),
                                  torch.tensor(scale_active),
                                  torch.tensor(scale_inactive))
        if interaction_pairs:
            ia = z[i_idx] * z[j_idx]
            scale_inter = torch.where(ia.bool(),
                                       torch.tensor(scale_active),
                                       torch.tensor(scale_inactive))
            prior_std = torch.cat([scale_main, scale_inter])
        else:
            prior_std = scale_main
        prior_var = prior_std ** 2
        prior_prec = torch.diag(1.0 / prior_var)

        XtX = X_full.T @ X_full
        post_prec = XtX / sigma2 + prior_prec
        try:
            L = torch.linalg.cholesky(post_prec)
            post_cov = torch.cholesky_inverse(L)
        except Exception:
            post_prec = post_prec + 1e-6 * torch.eye(d)
            L = torch.linalg.cholesky(post_prec)
            post_cov = torch.cholesky_inverse(L)
        post_mean = post_cov @ X_full.T @ y / sigma2
        theta = dist.MultivariateNormal(post_mean, post_cov).sample()

        residual = y - X_full @ theta
        rss = (residual * residual).sum()
        alpha_n = alpha0 + T / 2.0
        beta_n = beta0 + 0.5 * rss
        gamma_s = dist.Gamma(torch.tensor(alpha_n), torch.tensor(beta_n)).sample()
        sigma2 = 1.0 / gamma_s.clamp(min=1e-8)

        for k in range(p):
            z1 = z.clone(); z1[k] = 1.0
            z0 = z.clone(); z0[k] = 0.0

            def _student_marginal_for_z(z_try):
                sm = torch.where(z_try.bool(),
                                  torch.tensor(scale_active),
                                  torch.tensor(scale_inactive))
                if interaction_pairs:
                    ia_try = z_try[i_idx] * z_try[j_idx]
                    si = torch.where(ia_try.bool(),
                                      torch.tensor(scale_active),
                                      torch.tensor(scale_inactive))
                    prior_std_try = torch.cat([sm, si])
                else:
                    prior_std_try = sm
                prior_var_try = prior_std_try ** 2
                z_batch = z_try.unsqueeze(0).unsqueeze(0)
                y_batch = y.unsqueeze(0)
                xi_list_t = [X[t].unsqueeze(0) for t in range(T)]
                log_p = _log_py_given_z_student(
                    y_batch, xi_list_t, z_batch,
                    scale_active, scale_inactive,
                    alpha0, beta0,
                    interaction_pairs=interaction_pairs,
                ).squeeze()
                return log_p

            log_p1 = _student_marginal_for_z(z1) + math.log(pi_prior)
            log_p0 = _student_marginal_for_z(z0) + math.log(1.0 - pi_prior + 1e-10)
            log_total = torch.logaddexp(log_p1, log_p0)
            prob_active = torch.exp(log_p1 - log_total).clamp(0.0, 1.0)
            z[k] = torch.bernoulli(prob_active)

        if step >= burnin:
            z_samples.append(z.clone())

    z_samples = torch.stack(z_samples, dim=0)
    z_post = z_samples.float().mean(dim=0)
    z_pred = (z_post > threshold).float()
    return z_pred, z_post


def evaluate_myopic(
    num_factors,
    T,
    noise_scale=0.5,
    expected_active=0.2,
    scale_active=1.0,
    scale_inactive=0.01,
    sigma2_alpha0=3.0,
    sigma2_beta0=None,
    num_rollouts=50,
    num_gibbs_samples=200,
    burnin=100,
    threshold=0.5,
    seed=42,
    n_contrastive=100,
    n_y_samples=3,
    device="cpu",
    output_file=None,
):
    if sigma2_beta0 is None:
        sigma2_beta0 = (sigma2_alpha0 + 1.0) * (noise_scale ** 2)

    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    p = num_factors
    pi_prior = expected_active
    interaction_pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]

    candidates = enumerate_designs(p, device=device) if p <= 12 else None

    main_f1_list, main_tpr_list, main_fdr_list = [], [], []
    inter_f1_list, inter_tpr_list, inter_fdr_list = [], [], []

    print(f"\n{'=' * 64}")
    print("Myopic (SG-BOED-style) evaluation")
    print(f"{'=' * 64}")
    print(f"Dims:          p={p}, T={T}")
    print(f"Prior:         pi={pi_prior}, noise={noise_scale}")
    print(f"Rollouts:      {num_rollouts}")
    print(f"Candidates:    {'enumerated ({})'.format(2**p) if p <= 12 else 'random (2000)'}")
    print(f"Contrastive:   L={n_contrastive}, y-samples={n_y_samples}")
    print(f"Gibbs:         {num_gibbs_samples} samples + {burnin} burnin")
    print(f"Seed:          {seed}")
    print("-" * 64)

    for rollout_idx in range(num_rollouts):
        if rollout_idx % 5 == 0 or rollout_idx == num_rollouts - 1:
            print(f"  Rollout {rollout_idx + 1}/{num_rollouts}...", flush=True)

        true_z = torch.bernoulli(pi_prior * torch.ones(p))
        if true_z.sum() == 0:
            true_z[torch.randint(0, p, (1,))] = 1.0

        gamma_s = dist.Gamma(
            torch.tensor(sigma2_alpha0), torch.tensor(sigma2_beta0)
        ).sample()
        true_sigma2 = (1.0 / gamma_s.clamp(min=1e-8)).item()

        scale_main_t = torch.where(
            true_z.bool(),
            torch.tensor(scale_active),
            torch.tensor(scale_inactive),
        )
        true_theta_main = torch.randn(p) * scale_main_t

        if interaction_pairs:
            i_idx = torch.tensor([i for i, j in interaction_pairs])
            j_idx = torch.tensor([j for i, j in interaction_pairs])
            interact_active = true_z[i_idx] * true_z[j_idx]
            scale_interact_t = torch.where(
                interact_active.bool(),
                torch.tensor(scale_active),
                torch.tensor(scale_inactive),
            )
            true_theta_interact = torch.randn(len(interaction_pairs)) * scale_interact_t
        else:
            true_theta_interact = torch.tensor([])
            interact_active = torch.tensor([])

        cands = candidates if candidates is not None else random_candidates(p, 2000, device=device)
        X, y = run_myopic_rollout(
            true_z, true_theta_main, true_theta_interact, true_sigma2,
            T, p,
            scale_active, scale_inactive,
            sigma2_alpha0, sigma2_beta0,
            pi_prior, interaction_pairs,
            n_contrastive=n_contrastive,
            n_y_samples=n_y_samples,
            candidates=cands,
            device=device,
        )

        z_pred, _ = run_simple_gibbs(
            X, y, p, interaction_pairs,
            scale_active, scale_inactive,
            sigma2_alpha0, sigma2_beta0,
            pi_prior,
            num_samples=num_gibbs_samples, burnin=burnin,
            threshold=threshold,
        )

        true_main = true_z.bool()
        pred_main = z_pred.bool()
        tp = (pred_main & true_main).sum().item()
        fp = (pred_main & ~true_main).sum().item()
        fn = (~pred_main & true_main).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        fdr = fp / max(tp + fp, 1)
        main_f1_list.append(f1)
        main_tpr_list.append(recall)
        main_fdr_list.append(fdr)

        if interaction_pairs:
            pred_interact = torch.stack([
                pred_main[i] & pred_main[j] for i, j in interaction_pairs
            ])
            true_interact = interact_active.bool()
            itp = (pred_interact & true_interact).sum().item()
            ifp = (pred_interact & ~true_interact).sum().item()
            ifn = (~pred_interact & true_interact).sum().item()
            iprec = itp / max(itp + ifp, 1)
            irec = itp / max(itp + ifn, 1)
            if1 = 2 * iprec * irec / max(iprec + irec, 1e-10)
            ifdr = ifp / max(itp + ifp, 1)
            inter_f1_list.append(if1)
            inter_tpr_list.append(irec)
            inter_fdr_list.append(ifdr)

    def fmt(xs):
        xs = np.array(xs)
        return f"{xs.mean():.4f} +/- {xs.std() / np.sqrt(len(xs)):.4f}"

    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append(f"Myopic (SG-BOED) — p={p}, T={T}")
    lines.append("=" * 64)
    lines.append(f"Rollouts: {num_rollouts}")
    lines.append("")
    lines.append("Main Effects:")
    lines.append(f"  F1:          {fmt(main_f1_list)}")
    lines.append(f"  TPR:         {fmt(main_tpr_list)}")
    lines.append(f"  FDR:         {fmt(main_fdr_list)}")
    if inter_f1_list:
        lines.append("")
        lines.append("Interactions:")
        lines.append(f"  F1:          {fmt(inter_f1_list)}")
        lines.append(f"  TPR:         {fmt(inter_tpr_list)}")
        lines.append(f"  FDR:         {fmt(inter_fdr_list)}")
    lines.append("=" * 64)

    text = "\n".join(lines)
    print(text)
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            f.write(text + "\n")
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Myopic Bayesian (SG-BOED-style) baseline evaluation"
    )
    parser.add_argument("--num-factors", type=int, required=True)
    parser.add_argument("--num-experiments", type=int, required=True)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--expected-active", type=float, default=0.2)
    parser.add_argument("--scale-active", type=float, default=1.0)
    parser.add_argument("--scale-inactive", type=float, default=0.01)
    parser.add_argument("--sigma2-alpha0", type=float, default=3.0)
    parser.add_argument("--sigma2-beta0", type=float, default=None)
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gibbs-samples", type=int, default=200)
    parser.add_argument("--burnin", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-contrastive", type=int, default=100)
    parser.add_argument("--n-y-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    evaluate_myopic(
        num_factors=args.num_factors,
        T=args.num_experiments,
        noise_scale=args.noise_scale,
        expected_active=args.expected_active,
        scale_active=args.scale_active,
        scale_inactive=args.scale_inactive,
        sigma2_alpha0=args.sigma2_alpha0,
        sigma2_beta0=args.sigma2_beta0,
        num_rollouts=args.num_rollouts,
        num_gibbs_samples=args.num_gibbs_samples,
        burnin=args.burnin,
        threshold=args.threshold,
        seed=args.seed,
        n_contrastive=args.n_contrastive,
        n_y_samples=args.n_y_samples,
        device=args.device,
        output_file=args.output,
    )
    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")