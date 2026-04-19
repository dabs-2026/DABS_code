import argparse
import torch
import pyro
import pyro.distributions as dist
import math

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

def _student_log_marginal(y, X, z, interaction_pairs,
                           scale_active, scale_inactive,
                           alpha0, beta0):
    T, n_params = X.shape
    p = len(z)

    scale_main = torch.where(
        z.bool(),
        torch.tensor(scale_active),
        torch.tensor(scale_inactive)
    )
    if len(interaction_pairs) > 0:
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

    T          = len(designs)
    X          = build_design_matrix(designs, interaction_pairs)
    y          = observations
    num_params = p + num_interactions

    def sample_theta_given_z_sigma2(z, sigma2):
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

        prior_var  = prior_std ** 2
        prior_prec = torch.diag(1.0 / prior_var)

        XtX      = X.T @ X
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
        return theta, post_mean, post_cov

    def sample_sigma2_given_z_theta(z, theta, post_mean, post_cov):
        residual = y - X @ theta
        rss      = (residual * residual).sum()

        alpha_n = alpha0 + T / 2.0
        beta_n  = beta0  + 0.5 * rss

        gamma_sample = dist.Gamma(
            torch.tensor(alpha_n),
            torch.tensor(1.0 / beta_n)
        ).sample()
        sigma2 = 1.0 / gamma_sample.clamp(min=1e-8)
        return sigma2

    def sample_zk_given_rest(z, k, pi_val):
        z1 = z.clone(); z1[k] = 1.0
        z0 = z.clone(); z0[k] = 0.0

        log_p1 = (_student_log_marginal(y, X, z1, interaction_pairs,
                                         scale_active, scale_inactive,
                                         alpha0, beta0)
                  + math.log(pi_val))
        log_p0 = (_student_log_marginal(y, X, z0, interaction_pairs,
                                         scale_active, scale_inactive,
                                         alpha0, beta0)
                  + math.log(1.0 - pi_val + 1e-10))

        log_total   = torch.logaddexp(log_p1, log_p0)
        prob_active = torch.exp(log_p1 - log_total).clamp(0.0, 1.0)
        return torch.bernoulli(prob_active)

    sigma2_init = beta0 / (alpha0 - 1.0) if alpha0 > 1.0 else beta0 / alpha0
    sigma2 = torch.tensor(sigma2_init)

    z = dist.Bernoulli(pi * torch.ones(p)).sample()

    z_samples      = []
    theta_samples  = []
    sigma2_samples = []

    total_steps = burnin + num_samples
    print(f"\n  Running Gibbs sampler ({burnin} burnin + {num_samples} samples)...")
    print(f"  NIG prior: alpha0={alpha0:.2f}, beta0={beta0:.4f}  "
          f"(prior mode sigma={sigma2_init**0.5:.4f})")
    print(f"  [FIX] sigma2 update uses RSS only (fixed-variance spike-and-slab)")

    for step in range(total_steps):
        if step % 200 == 0:
            active = z.sum().int().item()
            print(f"    Step {step:>4}/{total_steps}  "
                  f"active={active}  sigma={sigma2.item()**0.5:.4f}")

        theta, post_mean, post_cov = sample_theta_given_z_sigma2(z, sigma2)

        sigma2 = sample_sigma2_given_z_theta(z, theta, post_mean, post_cov)

        for k in range(p):
            z[k] = sample_zk_given_rest(z, k, pi if not use_pi_hyperprior else pi)

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

    z_samples      = torch.stack(z_samples)
    theta_samples  = torch.stack(theta_samples)
    sigma2_samples = torch.stack(sigma2_samples)

    return z_samples, theta_samples, sigma2_samples


def report_results(true_z, true_theta_main, z_samples, theta_samples,
                   sigma2_samples, designs, observations,
                   interaction_pairs, threshold=0.5):

    p           = len(true_z)
    true_active = true_z.nonzero(as_tuple=True)[0].tolist()

    activity_probs   = z_samples.mean(0)
    predicted_active = (activity_probs > threshold).nonzero(as_tuple=True)[0].tolist()

    print("\n" + "=" * 60)
    print("SCREENING RESULTS")
    print("=" * 60)

    print(f"\nExperiments run: {len(designs)}")
    print(f"Factors screened: {p}")
    sigma2_mean = sigma2_samples.mean().item()
    sigma2_low  = sigma2_samples.quantile(0.025).item()
    sigma2_high = sigma2_samples.quantile(0.975).item()
    print(f"\nNoise variance posterior:")
    print(f"  E[sigma^2] = {sigma2_mean:.4f}  (sigma = {sigma2_mean**0.5:.4f})")
    print(f"  95% CI     = [{sigma2_low:.4f}, {sigma2_high:.4f}]")

    print("\nExperiment History:")
    print(f"  {'Step':>4}  {'Design':>22}  {'Response':>8}")
    for t in range(len(designs)):
        xi_str = "".join(["+" if x > 0 else "-" for x in designs[t][:20]])
        print(f"  {t+1:>4}  {xi_str:>22}  {observations[t].item():>8.3f}")

    print(f"\nActivity Probabilities  [threshold={threshold}]:")
    print(f"  {'Factor':>6}  {'P(active)':>10}  {'Predicted':>10}  {'True':>8}  {'OK':>4}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*4}")

    for k in range(p):
        prob      = activity_probs[k].item()
        predicted = prob > threshold
        true      = true_z[k].item() == 1
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

    def to1(idx_set):
        return sorted(i + 1 for i in idx_set)

    print(f"\nSummary (factor numbers are 1-based):")
    print(f"  True active:      {to1(true_set)}")
    print(f"  Predicted active: {to1(pred_set)}")
    print(f"  True positives:   {to1(true_pos)}")
    print(f"  False positives:  {to1(false_pos)}")
    print(f"  False negatives:  {to1(false_neg)}")
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  F1 score:  {f1:.2f}")

    delta = 1.0
    significant_active = [
        k for k in true_active if abs(true_theta_main[k].item()) > delta
    ]
    insignificant_active = [
        k for k in true_active if abs(true_theta_main[k].item()) <= delta
    ]
    if significant_active:
        sig_true_set  = set(significant_active)
        sig_true_pos  = sig_true_set & pred_set
        sig_false_pos = pred_set - sig_true_set - set(insignificant_active)
        sig_false_neg = sig_true_set - pred_set

        sig_precision = len(sig_true_pos) / max(len(sig_true_pos) + len(sig_false_pos), 1)
        sig_recall    = len(sig_true_pos) / max(len(sig_true_set), 1)
        sig_f1        = 2 * sig_precision * sig_recall / max(sig_precision + sig_recall, 1e-8)

        print(f"\nPractical Metrics (|θ| > {delta}):")
        print(f"  Significant active factors: {to1(set(significant_active))}")
        print(f"  True effect sizes:          "
              f"{[round(true_theta_main[k].item(), 3) for k in sorted(significant_active)]}")
        print(f"  Precision: {sig_precision:.2f}")
        print(f"  Recall:    {sig_recall:.2f}")
        print(f"  F1 score:  {sig_f1:.2f}")
    else:
        print(f"\nPractical Metrics (|θ| > {delta}): no significant active factors in this run")

    if true_active:
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
    sigma2_alpha0=3.0,
    sigma2_beta0=None,
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
    checkpoint    = torch.load(model_path, map_location="cpu")
    saved_params  = checkpoint.get("params", {})

    num_factors    = saved_params.get("num_factors",    num_factors)
    T              = saved_params.get("T",              T)
    encoding_dim   = saved_params.get("encoding_dim",   encoding_dim)
    hidden_dim     = saved_params.get("hidden_dim",     hidden_dim)
    noise_scale    = saved_params.get("noise_scale",    noise_scale)
    sigma2_alpha0  = saved_params.get("sigma2_alpha0",  sigma2_alpha0)
    sigma2_beta0   = saved_params.get("sigma2_beta0",   sigma2_beta0)
    encoder_type   = saved_params.get("encoder_type",   "original")
    scale_active   = saved_params.get("scale_active",   scale_active)
    scale_inactive = saved_params.get("scale_inactive", scale_inactive)
    pi_beta_a      = saved_params.get("pi_beta_a",      0.0)
    pi_beta_b      = saved_params.get("pi_beta_b",      0.0)
    expected_active_proportion = saved_params.get(
        "expected_active_proportion", expected_active_proportion
    )

    if sigma2_beta0 is None:
        sigma2_beta0 = (sigma2_alpha0 + 1.0) * (noise_scale ** 2)

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
        pi_beta_a=pi_beta_a,
        pi_beta_b=pi_beta_b,
        scale_active=scale_active,
        scale_inactive=scale_inactive,
        noise_scale=noise_scale,
        sigma2_alpha0=sigma2_alpha0,
        sigma2_beta0=sigma2_beta0,
    )
    screening_model.load_state_dict(checkpoint["model_state_dict"])
    screening_model.eval()
    print(f"  Loaded — {num_factors} factors, T={T}, encoder={encoder_type}")
    print(f"  NIG prior: alpha0={sigma2_alpha0}, beta0={sigma2_beta0:.4f}  "
          f"(prior mode sigma={((sigma2_beta0/(sigma2_alpha0+1))**0.5):.4f})")

    print(f"\nSampling true latent variables (ground truth)...")
    true_z = dist.Bernoulli(
        expected_active_proportion * torch.ones(num_factors)
    ).sample()
    if true_z.sum() == 0:
        true_z[torch.randint(0, num_factors, (1,))] = 1.0

    scale_main = torch.where(
        true_z.bool(), torch.tensor(scale_active), torch.tensor(scale_inactive)
    )
    true_theta_main = dist.Normal(torch.zeros(num_factors), scale_main).sample()

    interact_active = torch.stack([
        true_z[i] * true_z[j]
        for i, j in screening_model.interaction_pairs
    ], dim=-1)
    scale_interact = torch.where(
        interact_active.bool(),
        torch.tensor(scale_active), torch.tensor(scale_inactive)
    )
    true_theta_interact = dist.Normal(
        torch.zeros(screening_model.num_interactions), scale_interact
    ).sample()

    true_sigma2 = torch.tensor(sigma2_beta0 / (sigma2_alpha0 - 1.0)) if sigma2_alpha0 > 1 \
        else torch.tensor(sigma2_beta0 / sigma2_alpha0)

    true_active = true_z.nonzero(as_tuple=True)[0].tolist()
    print(f"  True active factors (1-based): {[k+1 for k in true_active]}")
    print(f"  True effect sizes:   "
          f"{true_theta_main[true_z.bool()].round(decimals=2).tolist()}")
    print(f"  True sigma (prior mean): {true_sigma2.item()**0.5:.4f}")

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

    z_samples, theta_samples, sigma2_samples = gibbs_sampler(
        designs, observations, screening_model,
        num_samples=num_gibbs_samples,
        burnin=burnin,
    )

    precision, recall, f1 = report_results(
        true_z=true_z,
        true_theta_main=true_theta_main,
        z_samples=z_samples,
        theta_samples=theta_samples,
        sigma2_samples=sigma2_samples,
        designs=designs,
        observations=observations,
        interaction_pairs=screening_model.interaction_pairs,
        threshold=threshold,
    )

    return precision, recall, f1


def evaluate_multi(
    model_path,
    num_rollouts=200,
    num_gibbs_samples=2000,
    burnin=500,
    threshold=0.5,
    seed=42,
):
    """Run multiple rollouts and aggregate F1, TPR, FDR, exact match."""
    import numpy as np

    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    checkpoint   = torch.load(model_path, map_location="cpu", weights_only=False)
    saved_params = checkpoint.get("params", {})

    num_factors    = saved_params.get("num_factors", 20)
    T              = saved_params.get("T", 10)
    encoding_dim   = saved_params.get("encoding_dim", 16)
    hidden_dim     = saved_params.get("hidden_dim", 64)
    noise_scale    = saved_params.get("noise_scale", 0.5)
    sigma2_alpha0  = saved_params.get("sigma2_alpha0", 3.0)
    sigma2_beta0   = saved_params.get("sigma2_beta0")
    encoder_type   = saved_params.get("encoder_type", "original")
    scale_active   = saved_params.get("scale_active", 1.0)
    scale_inactive = saved_params.get("scale_inactive", 0.01)
    pi_beta_a      = saved_params.get("pi_beta_a", 0.0)
    pi_beta_b      = saved_params.get("pi_beta_b", 0.0)
    expected_active = saved_params.get("expected_active_proportion", 0.2)

    if sigma2_beta0 is None:
        sigma2_beta0 = (sigma2_alpha0 + 1.0) * (noise_scale ** 2)

    design_network_type = saved_params.get("design_network_type", "dad")

    if design_network_type == "static":
        from modules import BatchDesignBaseline
        design_net = BatchDesignBaseline(T=T, design_dim=num_factors)
    elif design_network_type == "random":
        from modules import RandomDesignBaseline
        design_net = RandomDesignBaseline(design_dim=num_factors)
    else:
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
        expected_active_proportion=expected_active,
        pi_beta_a=pi_beta_a,
        pi_beta_b=pi_beta_b,
        scale_active=scale_active,
        scale_inactive=scale_inactive,
        noise_scale=noise_scale,
        sigma2_alpha0=sigma2_alpha0,
        sigma2_beta0=sigma2_beta0,
    )
    # Register static_floor buffers if the checkpoint has them
    # (so load_state_dict doesn't fail with "Unexpected key")
    sd = checkpoint["model_state_dict"]
    floor_keys = [k for k in sd.keys() if "static_floor_" in k]
    if floor_keys and design_network_type == "dad":
        for k in floor_keys:
            buf_name = k.replace("design_net.", "")
            screening_model.design_net.register_buffer(buf_name, sd[k].clone())
        screening_model.design_net._has_static_floor = True
        screening_model.design_net._n_static = len(floor_keys)

    screening_model.load_state_dict(sd, strict=False)
    screening_model.eval()

    print("=" * 64)
    print("Multi-rollout Gibbs evaluation")
    print("=" * 64)
    print(f"Checkpoint:    {model_path}")
    print(f"Policy type:   {saved_params.get('design_network_type', 'n/a')}")
    print(f"Encoder:       {encoder_type}")
    print(f"Dims:          p={num_factors}, T={T}")
    print(f"Prior:         pi={expected_active}, noise={noise_scale}")
    print(f"Rollouts:      {num_rollouts}")
    print(f"Gibbs:         {num_gibbs_samples} samples + {burnin} burnin")
    print(f"Seed:          {seed}")
    print("-" * 64)

    f1_list, tpr_list, fdr_list, exact_list = [], [], [], []

    for r in range(num_rollouts):
        if r % 20 == 0:
            print(f"  Rollout {r+1}/{num_rollouts}...")

        # Sample ground truth
        true_z = dist.Bernoulli(
            expected_active * torch.ones(num_factors)
        ).sample()

        scale_main = torch.where(
            true_z.bool(), torch.tensor(scale_active), torch.tensor(scale_inactive)
        )
        true_theta_main = dist.Normal(torch.zeros(num_factors), scale_main).sample()

        interact_active = torch.stack([
            true_z[i] * true_z[j]
            for i, j in screening_model.interaction_pairs
        ], dim=-1)
        scale_interact = torch.where(
            interact_active.bool(),
            torch.tensor(scale_active), torch.tensor(scale_inactive)
        )
        true_theta_interact = dist.Normal(
            torch.zeros(screening_model.num_interactions), scale_interact
        ).sample()

        # Run policy
        with torch.no_grad():
            designs, observations = run_policy(
                screening_model, true_z, true_theta_main, true_theta_interact
            )

        # Gibbs sampler (quiet mode)
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        z_samples, theta_samples, sigma2_samples = gibbs_sampler(
            designs, observations, screening_model,
            num_samples=num_gibbs_samples,
            burnin=burnin,
        )
        sys.stdout = old_stdout

        # Compute metrics
        activity_probs = z_samples.mean(0)
        predicted = (activity_probs > threshold)
        true = true_z.bool()

        tp = (predicted & true).sum().item()
        fp = (predicted & ~true).sum().item()
        fn = (~predicted & true).sum().item()

        tpr = tp / max(tp + fn, 1)
        fdr = fp / max(tp + fp, 1)

        if (tp + fp + fn) > 0:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        else:
            f1 = 1.0 if true.sum() == 0 else 0.0

        exact = float(predicted.equal(true))

        f1_list.append(f1)
        tpr_list.append(tpr)
        fdr_list.append(fdr)
        exact_list.append(exact)

    # Aggregate
    f1_arr = np.array(f1_list)
    tpr_arr = np.array(tpr_list)
    fdr_arr = np.array(fdr_list)
    exact_arr = np.array(exact_list)

    def fmt(arr):
        return f"{arr.mean():.4f} +/- {arr.std()/np.sqrt(len(arr)):.4f}"

    print()
    print("Results:")
    print(f"  F1:          {fmt(f1_arr)}")
    print(f"  TPR:         {fmt(tpr_arr)}")
    print(f"  FDR:         {fmt(fdr_arr)}")
    print(f"  Exact Match: {fmt(exact_arr)}")
    print(f"  Rollouts:    {num_rollouts}")
    print("=" * 64)

    return {
        "f1": f1_arr, "tpr": tpr_arr,
        "fdr": fdr_arr, "exact": exact_arr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained DAD screening policy"
    )
    parser.add_argument("--model-path",         required=True,  type=str)
    parser.add_argument("--num-rollouts",       default=1,      type=int,
                        help="Number of rollouts. Use 1 for detailed single-run output, "
                             "200+ for aggregated metrics.")
    parser.add_argument("--num-gibbs-samples",  default=2000,   type=int)
    parser.add_argument("--burnin",             default=500,    type=int)
    parser.add_argument("--threshold",          default=0.5,    type=float)
    parser.add_argument("--seed",               default=42,     type=int)

    # Legacy single-run args (used when num_rollouts=1)
    parser.add_argument("--num-factors",        default=20,     type=int)
    parser.add_argument("--num-experiments",    default=10,     type=int)
    parser.add_argument("--expected-active",    default=0.2,    type=float)
    parser.add_argument("--scale-active",       default=1.0,    type=float)
    parser.add_argument("--scale-inactive",     default=0.01,   type=float)
    parser.add_argument("--noise-scale",        default=0.5,    type=float)
    parser.add_argument("--sigma2-alpha0",      default=3.0,    type=float)
    parser.add_argument("--sigma2-beta0",       default=None,   type=float)
    parser.add_argument("--encoding-dim",       default=16,     type=int)
    parser.add_argument("--hidden-dim",         default=64,     type=int)

    args = parser.parse_args()

    if args.num_rollouts > 1:
        evaluate_multi(
            model_path=args.model_path,
            num_rollouts=args.num_rollouts,
            num_gibbs_samples=args.num_gibbs_samples,
            burnin=args.burnin,
            threshold=args.threshold,
            seed=args.seed,
        )
    else:
        evaluate(
            model_path=args.model_path,
            num_factors=args.num_factors,
            T=args.num_experiments,
            expected_active_proportion=args.expected_active,
            scale_active=args.scale_active,
            scale_inactive=args.scale_inactive,
            noise_scale=args.noise_scale,
            sigma2_alpha0=args.sigma2_alpha0,
            sigma2_beta0=args.sigma2_beta0,
            encoding_dim=args.encoding_dim,
            hidden_dim=args.hidden_dim,
            num_gibbs_samples=args.num_gibbs_samples,
            burnin=args.burnin,
            threshold=args.threshold,
            seed=args.seed,
        )