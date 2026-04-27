import argparse, os, sys, time, csv
import numpy as np
import torch

PB_GENERATORS = {
    8:  [+1, +1, +1, -1, +1, -1, -1],
    12: [+1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1],
    16: [+1, +1, +1, +1, -1, +1, -1, +1, +1, -1, -1, +1, -1, -1, -1],
    20: [+1, +1, -1, -1, +1, +1, +1, +1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, -1],
    24: [+1, +1, +1, +1, +1, -1, +1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, -1, +1, -1, -1, -1, -1],
}

def pb_matrix(T):
    if T not in PB_GENERATORS:
        raise ValueError(f"No PB generator for T={T}. Available: {sorted(PB_GENERATORS.keys())}")
    g = np.array(PB_GENERATORS[T])
    n_cols = T - 1
    M = np.zeros((T, n_cols), dtype=int)
    for i in range(T - 1):
        M[i] = np.roll(g, i)
    M[T - 1] = -1
    return M


def sample_truth(num_factors, pi, tau_active, tau_inactive, sigma2_alpha, sigma2_beta, rng):
    z = rng.binomial(1, pi, size=num_factors).astype(np.float32)
    scales = np.where(z > 0.5, tau_active, tau_inactive)
    beta = rng.normal(0.0, scales)
    sigma2 = sigma2_beta / rng.gamma(shape=sigma2_alpha)
    return z, beta, sigma2


def forward_response(xi_row, beta, sigma2, rng):
    mean = xi_row @ beta
    return mean + rng.normal(0.0, np.sqrt(sigma2))


def gibbs_posterior(X, y, pi, tau_active, tau_inactive, sigma2_est,
                    G=200, burnin=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    T, p = X.shape
    z = (rng.random(p) < pi).astype(np.float32)
    tau2_vec = np.where(z > 0.5, tau_active**2, tau_inactive**2)
    XtX = X.T @ X
    Xty = X.T @ y

    z_samples = np.zeros((G + burnin, p))
    beta_samples = np.zeros((G + burnin, p))

    for g in range(G + burnin):
        tau2_vec = np.where(z > 0.5, tau_active**2, tau_inactive**2)
        prec = XtX / sigma2_est + np.diag(1.0 / tau2_vec)
        cov = np.linalg.inv(prec + 1e-6 * np.eye(p))
        mean = cov @ (Xty / sigma2_est)
        beta = rng.multivariate_normal(mean, cov)

        for k in range(p):
            log_odds = (np.log(pi / (1 - pi))
                        + 0.5 * np.log(tau_inactive**2 / tau_active**2)
                        - 0.5 * beta[k]**2 * (1.0 / tau_active**2 - 1.0 / tau_inactive**2))
            prob1 = 1.0 / (1.0 + np.exp(-log_odds))
            z[k] = 1.0 if rng.random() < prob1 else 0.0

        z_samples[g] = z
        beta_samples[g] = beta

    z_post = z_samples[burnin:].mean(axis=0)
    beta_post = beta_samples[burnin:].mean(axis=0)
    return z_post, beta_post


def f1_tpr_fdr(z_true, z_pred):
    tp = int(((z_pred == 1) & (z_true == 1)).sum())
    fp = int(((z_pred == 1) & (z_true == 0)).sum())
    fn = int(((z_pred == 0) & (z_true == 1)).sum())
    if tp + fp == 0:
        prec = 0.0; fdr = 0.0
    else:
        prec = tp / (tp + fp)
        fdr = fp / (tp + fp)
    if tp + fn == 0:
        tpr = 0.0
    else:
        tpr = tp / (tp + fn)
    if prec + tpr == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * tpr / (prec + tpr)
    return f1, tpr, fdr


def eig_mc(X, z_true, beta_true, sigma2_true, pi, tau_active, tau_inactive,
           L=500, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    T, p = X.shape
    y = X @ beta_true + rng.normal(0, np.sqrt(sigma2_true), size=T)
    def log_marg(z):
        tau2_vec = np.where(z > 0.5, tau_active**2, tau_inactive**2)
        Sigma = X @ np.diag(tau2_vec) @ X.T + sigma2_true * np.eye(T)
        sign, logdet = np.linalg.slogdet(Sigma + 1e-4 * np.eye(T))
        return -0.5 * (T * np.log(2 * np.pi) + logdet
                       + y @ np.linalg.solve(Sigma + 1e-4 * np.eye(T), y))
    log_num = log_marg(z_true)
    log_contrast = [log_num]
    for _ in range(L):
        z_i = (rng.random(p) < pi).astype(float)
        log_contrast.append(log_marg(z_i))
    log_contrast = np.array(log_contrast)
    log_mean = np.log(L + 1) * (-1) + np.log(np.exp(log_contrast - log_contrast.max()).sum()) + log_contrast.max()
    return log_num - log_mean


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--num-outer-samples", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    p, T = args.num_factors, args.num_experiments
    pi = args.expected_active
    tau_a = args.scale_active
    tau_i = args.scale_inactive

    if args.sigma2_beta0 is None:
        args.sigma2_beta0 = (args.sigma2_alpha0 + 1.0) * (args.noise_scale ** 2)

    if T not in PB_GENERATORS:
        raise SystemExit(f"Error: no PB design available for T={T}. Valid: {sorted(PB_GENERATORS.keys())}")
    M = pb_matrix(T)
    if p > M.shape[1]:
        raise SystemExit(f"Error: PB-{T} supports at most {M.shape[1]} factors (you requested p={p}).")
    X_pb = M[:, :p].astype(np.float32)

    print(f"Plackett-Burman evaluation")
    print(f"  p={p}, T={T}")
    print(f"  Prior: pi={pi}, noise={args.noise_scale}, tau_active={tau_a}, tau_inactive={tau_i}")
    print(f"  Rollouts: {args.num_rollouts}, Gibbs: {args.num_gibbs_samples}+{args.burnin}")
    print(f"  PB design matrix ({T}x{p}):")
    print(X_pb.astype(int))

    rng = np.random.default_rng(args.seed)

    f1_list, tpr_list, fdr_list, eig_list = [], [], [], []
    for rollout in range(args.num_rollouts):
        z_true, beta_true, sigma2_true = sample_truth(
            p, pi, tau_a, tau_i, args.sigma2_alpha0, args.sigma2_beta0, rng
        )
        y = np.array([forward_response(X_pb[t], beta_true, sigma2_true, rng)
                      for t in range(T)], dtype=np.float32)

        z_post, beta_post = gibbs_posterior(
            X_pb, y, pi, tau_a, tau_i, sigma2_true,
            G=args.num_gibbs_samples, burnin=args.burnin, rng=rng
        )
        z_pred = (z_post > 0.5).astype(int)
        f1, tpr, fdr = f1_tpr_fdr(z_true.astype(int), z_pred)
        f1_list.append(f1); tpr_list.append(tpr); fdr_list.append(fdr)

        if rollout < 10:
            try:
                eig_list.append(eig_mc(X_pb, z_true, beta_true, sigma2_true,
                                        pi, tau_a, tau_i, L=args.num_outer_samples, rng=rng))
            except Exception:
                pass

        if (rollout + 1) % 10 == 0:
            print(f"  [{rollout+1}/{args.num_rollouts}] F1 so far = {np.mean(f1_list):.4f}")

    def stat(xs):
        xs = np.array(xs)
        return xs.mean(), xs.std() / np.sqrt(len(xs))

    f1_m, f1_se = stat(f1_list)
    tpr_m, tpr_se = stat(tpr_list)
    fdr_m, fdr_se = stat(fdr_list)
    eig_m, eig_se = (stat(eig_list) if eig_list else (np.nan, np.nan))

    report = f"""
=== Plackett-Burman Baseline ===
Dims:          p={p}, T={T}
Prior:         pi={pi}, noise={args.noise_scale}
Rollouts:      {args.num_rollouts}
Gibbs:         {args.num_gibbs_samples} samples + {args.burnin} burnin
Seed:          {args.seed}
----------------------------------------------------------------
  F1:          {f1_m:.4f} +/- {f1_se:.4f}
  TPR:         {tpr_m:.4f} +/- {tpr_se:.4f}
  FDR:         {fdr_m:.4f} +/- {fdr_se:.4f}
  EIG:         {eig_m:.4f} +/- {eig_se:.4f}  (n={len(eig_list)})
"""
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()