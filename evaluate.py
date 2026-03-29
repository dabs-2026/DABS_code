import argparse
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import numpy as np

from modules import build_screening_network
from screening_model import ScreeningExperiment


def run_policy(screening_model, true_z, true_theta_main, true_theta_interact):
    true_theta = {
        "z": true_z,
        "theta_main": true_theta_main,
        "theta_interact": true_theta_interact,
    }
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


def gibbs_sampler(
    designs,
    observations,
    screening_model,
    num_samples=2000,
    burnin=500,
):
    p = screening_model.num_factors
    interaction_pairs = screening_model.interaction_pairs
    num_interactions = len(interaction_pairs)
    pi = screening_model.pi_fixed
    scale_active = screening_model.scale_active
    scale_inactive = screening_model.scale_inactive
    sigma2 = screening_model.noise_scale.item() ** 2

    T = len(designs)
    X = build_design_matrix(designs, interaction_pairs)
    y = observations
    num_params = p + num_interactions

    def sample_theta_given_z(z):
        scale_main = torch.where(
            z.bool(),
            torch.tensor(scale_active),
            torch.tensor(scale_inactive)
        )
        if num_interactions > 0:
            interact_active = torch.stack([
                z[i] * z[j] for i, j in interaction_pairs
            ], dim=-1)
            scale_interact = torch.where(
                interact_active.bool(),
                torch.tensor(scale_active),
                torch.tensor(scale_inactive)
            )
            prior_std = torch.cat([scale_main, scale_interact])
        else:
            prior_std = scale_main

        prior_var = prior_std ** 2

        XtX = X.T @ X
        prior_prec = torch.diag(1.0 / prior_var)
        post_prec = XtX / sigma2 + prior_prec

        try:
            L = torch.linalg.cholesky(post_prec)
            post_cov = torch.cholesky_inverse(L)
        except Exception:
            post_prec = post_prec + 1e-6 * torch.eye(num_params)
            L = torch.linalg.cholesky(post_prec)
            post_cov = torch.cholesky_inverse(L)

        post_mean = post_cov @ X.T @ y / sigma2

        theta = dist.MultivariateNormal(post_mean, post_cov).sample()
        return theta

    def compute_log_marginal_likelihood(z):
        scale_main = torch.where(
            z.bool(),
            torch.tensor(scale_active),
            torch.tensor(scale_inactive)
        )
        if num_interactions > 0:
            interact_active = torch.stack([
                z[i] * z[j] for i, j in interaction_pairs
            ], dim=-1)
            scale_interact = torch.where(
                interact_active.bool(),
                torch.tensor(scale_active),
                torch.tensor(scale_inactive)
            )
            prior_std = torch.cat([scale_main, scale_interact])
        else:
            prior_std = scale_main

        prior_var = prior_std ** 2
        cov = (X * prior_var.unsqueeze(0)) @ X.T + sigma2 * torch.eye(T)
        try:
            log_lik = dist.MultivariateNormal(
                torch.zeros(T), covariance_matrix=cov
            ).log_prob(y)
        except Exception:
            log_lik = torch.tensor(-1e10)
        return log_lik

    def sample_zk_given_rest(z, k):
        z1 = z.clone()
        z1[k] = 1.0
        log_p1 = compute_log_marginal_likelihood(z1) + torch.log(torch.tensor(pi))

        z0 = z.clone()
        z0[k] = 0.0
        log_p0 = compute_log_marginal_likelihood(z0) + torch.log(torch.tensor(1.0 - pi))

        log_total = torch.logaddexp(log_p1, log_p0)
        prob_active = torch.exp(log_p1 - log_total)

        return torch.bernoulli(prob_active)

    z = dist.Bernoulli(pi * torch.ones(p)).sample()

    z_samples = []
    theta_samples = []

    total_steps = burnin + num_samples
    print(f"\n  Running Gibbs sampler ({burnin} burnin + {num_samples} samples)...")

    for step in range(total_steps):
        if step % 200 == 0:
            active = z.sum().int().item()
            print(f"    Step {step:>4}/{total_steps}  active factors: {active}")

        theta = sample_theta_given_z(z)

        for k in range(p):
            z[k] = sample_zk_given_rest(z, k)

        if step >= burnin:
            z_samples.append(z.clone())
            theta_samples.append(theta.clone())

    z_samples = torch.stack(z_samples)
    theta_samples = torch.stack(theta_samples)

    return z_samples, theta_samples


def report_results(true_z, true_theta_main, z_samples, theta_samples,
                   designs, observations, interaction_pairs, threshold=0.5):

    p = len(true_z)
    true_active = true_z.nonzero(as_tuple=True)[0].tolist()

    activity_probs = z_samples.mean(0)
    predicted_active = (activity_probs > threshold).nonzero(as_tuple=True)[0].tolist()

    print("\n" + "=" * 60)
    print("SCREENING RESULTS")
    print("=" * 60)

    print(f"\nExperiments run: {len(designs)}")
    print(f"Factors screened: {p}")

    print("\nExperiment History:")
    print(f"  {'Step':>4}  {'Design':>22}  {'Response':>8}")
    for t in range(len(designs)):
        xi_str = "".join(["+" if x > 0 else "-" for x in designs[t][:20]])
        print(f"  {t+1:>4}  {xi_str:>22}  {observations[t].item():>8.3f}")

    print(f"\nActivity Probabilities  [threshold={threshold}]:")
    print(f"  {'Factor':>6}  {'P(active)':>10}  {'Predicted':>10}  {'True':>8}  {'OK':>4}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*4}")

    for k in range(p):
        prob = activity_probs[k].item()
        predicted = prob > threshold
        true = true_z[k].item() == 1
        if true or predicted or prob > 0.1:
            marker = "✓" if (predicted == true) else "✗"
            print(
                f"  {k+1:>6}  {prob:>10.3f}  "
                f"{'ACTIVE' if predicted else 'inactive':>10}  "
                f"{'ACTIVE' if true else 'inactive':>8}  "
                f"{marker:>4}"
            )

    true_set  = set(true_active)
    pred_set  = set(predicted_active)
    true_pos  = true_set & pred_set
    false_pos = pred_set - true_set
    false_neg = true_set - pred_set

    precision = len(true_pos) / max(len(pred_set), 1)
    recall    = len(true_pos) / max(len(true_set), 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\nSummary:")
    print(f"  True active:      {sorted(true_active)}")
    print(f"  Predicted active: {sorted(predicted_active)}")
    print(f"  True positives:   {sorted(true_pos)}")
    print(f"  False positives:  {sorted(false_pos)}")
    print(f"  False negatives:  {sorted(false_neg)}")
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  F1 score:  {f1:.2f}")

    delta = 1.0
    significant_active = [
        k for k in true_active
        if abs(true_theta_main[k].item()) > delta
    ]
    insignificant_active = [
        k for k in true_active
        if abs(true_theta_main[k].item()) <= delta
    ]
    if len(significant_active) > 0:
        sig_true_set  = set(significant_active)
        sig_true_pos  = sig_true_set & pred_set
        sig_false_pos = pred_set - sig_true_set - set(insignificant_active)
        sig_false_neg = sig_true_set - pred_set

        sig_precision = len(sig_true_pos) / max(len(sig_true_pos) + len(sig_false_pos), 1)
        sig_recall    = len(sig_true_pos) / max(len(sig_true_set), 1)
        sig_f1        = 2 * sig_precision * sig_recall / max(sig_precision + sig_recall, 1e-8)

        print(f"\nPractical Metrics (|θ| > {delta}):")
        print(f"  Significant active factors: {[k+1 for k in sorted(significant_active)]}")
        print(f"  True effect sizes:          {[round(true_theta_main[k].item(),3) for k in sorted(significant_active)]}")
        print(f"  Precision: {sig_precision:.2f}")
        print(f"  Recall:    {sig_recall:.2f}")
        print(f"  F1 score:  {sig_f1:.2f}")
    else:
        print(f"\nPractical Metrics (|θ| > {delta}): no significant active factors in this run")

    if len(true_active) > 0:
        print(f"\nEffect Size Estimates (truly active factors):")
        print(f"  {'Factor':>6}  {'True θ':>8}  {'Post.mean':>10}  {'95% CI':>22}  {'OK':>4}")
        for k in true_active:
            true_val  = true_theta_main[k].item()
            post_mean = theta_samples[:, k].mean().item()
            post_low  = theta_samples[:, k].quantile(0.025).item()
            post_high = theta_samples[:, k].quantile(0.975).item()
            in_ci     = "✓" if post_low <= true_val <= post_high else "✗"
            print(
                f"  {k+1:>6}  {true_val:>8.3f}  {post_mean:>10.3f}  "
                f"[{post_low:>6.3f}, {post_high:>6.3f}]  {in_ci}"
            )

    print("=" * 60)
    return precision, recall, f1


def evaluate(
    model_path,
    num_factors=20,
    T=10,
    expected_active_proportion=0.2,
    scale_active=1.0,
    scale_inactive=0.01,
    noise_scale=0.5,
    encoding_dim=16,
    hidden_dim=64,
    num_gibbs_samples=2000,
    burnin=500,
    threshold=0.5,
    seed=-1,
):
    if seed == -1:
        seed = torch.randint(0, 10000, (1,)).item()
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    print(f"Seed: {seed}")

    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")
    saved_params = checkpoint.get("params", {})

    num_factors   = saved_params.get("num_factors",   num_factors)
    T             = saved_params.get("T",             T)
    encoding_dim  = saved_params.get("encoding_dim",  encoding_dim)
    hidden_dim    = saved_params.get("hidden_dim",    hidden_dim)
    noise_scale   = saved_params.get("noise_scale",   noise_scale)
    encoder_type  = saved_params.get("encoder_type",  "original")
    expected_active_proportion = saved_params.get(
        "expected_active_proportion", expected_active_proportion
    )

    design_net = build_screening_network(
        num_factors=num_factors,
        encoding_dim=encoding_dim,
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
    )
    screening_model = ScreeningExperiment(
        design_net=design_net,
        num_factors=num_factors,
        T=T,
        expected_active_proportion=expected_active_proportion,
        scale_active=scale_active,
        scale_inactive=scale_inactive,
        noise_scale=noise_scale,
    )
    screening_model.load_state_dict(checkpoint["model_state_dict"])
    screening_model.eval()
    print(f"  Loaded — {num_factors} factors, T={T}, encoder={encoder_type}")

    print(f"\nSampling true latent variables (ground truth)...")
    true_z = dist.Bernoulli(
        expected_active_proportion * torch.ones(num_factors)
    ).sample()

    if true_z.sum() == 0:
        true_z[torch.randint(0, num_factors, (1,))] = 1.0

    scale_main = torch.where(
        true_z.bool(),
        torch.tensor(scale_active),
        torch.tensor(scale_inactive)
    )
    true_theta_main = dist.Normal(
        torch.zeros(num_factors), scale_main
    ).sample()

    interact_active = torch.stack([
        true_z[i] * true_z[j]
        for i, j in screening_model.interaction_pairs
    ], dim=-1)
    scale_interact = torch.where(
        interact_active.bool(),
        torch.tensor(scale_active),
        torch.tensor(scale_inactive)
    )
    true_theta_interact = dist.Normal(
        torch.zeros(screening_model.num_interactions), scale_interact
    ).sample()

    true_active = true_z.nonzero(as_tuple=True)[0].tolist()
    print(f"  True active factors: {true_active}")
    print(f"  True effect sizes:   "
          f"{true_theta_main[true_z.bool()].round(decimals=2).tolist()}")

    print(f"\nRunning DAD policy for T={T} experiments...")
    with torch.no_grad():
        designs, observations = run_policy(
            screening_model, true_z, true_theta_main, true_theta_interact
        )
    print(f"  Done")

    unique_designs = len(set([tuple(d.tolist()) for d in designs]))
    print(f"  Unique designs used: {unique_designs}/{T}")
    if unique_designs <= 2:
        print(f"  WARNING: policy is not adapting well — consider retraining")

    z_samples, theta_samples = gibbs_sampler(
        designs, observations, screening_model,
        num_samples=num_gibbs_samples,
        burnin=burnin,
    )

    precision, recall, f1 = report_results(
        true_z=true_z,
        true_theta_main=true_theta_main,
        z_samples=z_samples,
        theta_samples=theta_samples,
        designs=designs,
        observations=observations,
        interaction_pairs=screening_model.interaction_pairs,
        threshold=threshold,
    )

    return precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained DAD screening policy"
    )
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--num-factors", default=20, type=int)
    parser.add_argument("--num-experiments", default=10, type=int)
    parser.add_argument("--expected-active", default=0.2, type=float)
    parser.add_argument("--scale-active", default=1.0, type=float)
    parser.add_argument("--scale-inactive", default=0.01, type=float)
    parser.add_argument("--noise-scale", default=0.5, type=float)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--hidden-dim", default=64, type=int)
    parser.add_argument("--num-gibbs-samples", default=2000, type=int)
    parser.add_argument("--burnin", default=500, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--seed", default=-1, type=int)

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        num_factors=args.num_factors,
        T=args.num_experiments,
        expected_active_proportion=args.expected_active,
        scale_active=args.scale_active,
        scale_inactive=args.scale_inactive,
        noise_scale=args.noise_scale,
        encoding_dim=args.encoding_dim,
        hidden_dim=args.hidden_dim,
        num_gibbs_samples=args.num_gibbs_samples,
        burnin=args.burnin,
        threshold=args.threshold,
        seed=args.seed,
    )