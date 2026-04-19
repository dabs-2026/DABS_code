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

        XtX       = X.T @ X
        post_prec = XtX / sigma2 + prior_prec
        try:
            L        = torch.linalg.cholesky(post_prec)
            post_cov = torch.cholesky_inverse(L)
        except Exception:
            post_prec = post_prec + 1e-6 * torch.eye(num_params)
            L         = torch.linalg.cholesky(post_prec)
            post_cov  = torch.cholesky_inverse(L)

        post_mean = post_cov @ X.T @ y / sigma2
        theta = dist.MultivariateNormal(
            post_mean, sigma2 * post_cov
        ).sample()
        return theta

    def sample_sigma2_given_z_theta(z, theta):
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

        residual   = y - X @ theta
        rss        = (residual * residual).sum()
        prior_quad = (theta * theta / prior_var).sum()

        alpha_n = alpha0 + T / 2.0
        beta_n  = beta0 + 0.5 * rss + 0.5 * prior_quad

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

    sigma2_init = beta0 / (alpha0 + 1.0)
    sigma2 = torch.tensor(sigma2_init)

    z = dist.Bernoulli(pi * torch.ones(p)).sample()

    z_samples      = []
    theta_samples  = []
    sigma2_samples = []

    total_steps = burnin + num_samples
    print(f"\n  Running Gibbs sampler ({burnin} burnin + {num_samples} samples)...")
    print(f"  NIG prior: alpha0={alpha0:.2f}, beta0={beta0:.4f}  "
          f"(prior mode sigma={sigma2_init**0.5:.4f})")

    for step in range(total_steps):
        if step % 200 == 0:
            active = z.sum().int().item()
            print(f"    Step {step:>4}/{total_steps}  "
                  f"active={active}  sigma={sigma2.item()**0.5:.4f}")

        theta = sample_theta_given_z_sigma2(z, sigma2)

        sigma2 = sample_sigma2_given_z_theta(z, theta)

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

    print("\n" + "=" * 70)
    print("SCREENING RESULTS (MAIN EFFECTS + INTERACTIONS)")
    print("=" * 70)

    print(f"\nExperiments run: {len(designs)}")
    print(f"Factors screened: {p}")

    sigma2_mean = sigma2_samples.mean().item()
    sigma2_low  = sigma2_samples.quantile(0.025).item()
    sigma2_high = sigma2_samples.quantile(0.975).item()
    print(f"\nNoise variance posterior:")
    print(f"  E[sigma^2] = {sigma2_mean:.4f}  (sigma = {sigma2_mean**0.5:.4f})")
    print(f"  95% CI     = [{sigma2_low:.4f}, {sigma2_high:.4f}]")

    print("\nExperiment History:")
    for t in range(len(designs)):
        xi_str = "".join(["+" if x > 0 else "-" for x in designs[t][:20]])
        print(f"  {t+1:>3}   {xi_str:>25}   {observations[t].item():8.3f}")

    print(f"\nMain Effect Activity Probabilities [threshold={threshold}]:")
    print(f"  {'Factor':>6}  {'P(active)':>10}  {'Predicted':>10}  {'True':>8}  {'OK':>4}")
    for k in range(p):
        prob      = activity_probs[k].item()
        predicted = prob > threshold
        true_val  = bool(true_z[k])
        if true_val or predicted or prob > 0.1:
            marker = "✓" if (predicted == true_val) else "✗"
            print(f"  {k+1:>6}  {prob:>10.3f}  "
                  f"{'ACTIVE' if predicted else 'inactive':>10}  "
                  f"{'ACTIVE' if true_val else 'inactive':>8}  {marker:>4}")

    if theta_samples.shape[1] > p and len(interaction_pairs) > 0:
        print(f"\nInteraction Activity Probabilities [threshold={threshold}]:")
        print(f"  {'Pair':>8}  {'P(active)':>10}  {'Predicted':>10}  {'True':>8}  {'OK':>4}")
        for idx, (i, j) in enumerate(interaction_pairs):
            prob_interact = (theta_samples[:, p + idx].abs() > 0.1).float().mean().item()
            true_interact = bool(true_z[i] * true_z[j])
            predicted     = prob_interact > threshold
            if true_interact or predicted or prob_interact > 0.1:
                marker = "✓" if predicted == true_interact else "✗"
                print(f"  {i+1:>2}x{j+1:<2}  {prob_interact:>10.3f}  "
                      f"{'ACTIVE' if predicted else 'inactive':>10}  "
                      f"{'ACTIVE' if true_interact else 'inactive':>8}  {marker:>4}")

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
    print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print("=" * 70)

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
    checkpoint   = torch.load(model_path, map_location="cpu")
    saved_params = checkpoint.get("params", {})

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

    interaction_pairs = screening_model.interaction_pairs
    interact_active = torch.stack([
        true_z[i] * true_z[j] for i, j in interaction_pairs
    ], dim=-1)
    scale_interact = torch.where(
        interact_active.bool(),
        torch.tensor(scale_active), torch.tensor(scale_inactive)
    )
    true_theta_interact = dist.Normal(
        torch.zeros(len(interaction_pairs)), scale_interact
    ).sample()

    true_sigma2 = dist.InverseGamma(
        torch.tensor(sigma2_alpha0),
        torch.tensor(sigma2_beta0)
    ).sample() if hasattr(dist, "InverseGamma") else torch.tensor(
        sigma2_beta0 / (sigma2_alpha0 - 1.0)
    )

    true_active = true_z.nonzero(as_tuple=True)[0].tolist()
    print(f"  True active factors: {true_active}")
    print(f"  True effect sizes:   "
          f"{true_theta_main[true_z.bool()].round(decimals=2).tolist()}")
    print(f"  True sigma:          {true_sigma2.item()**0.5:.4f}")

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
        interaction_pairs=interaction_pairs,
        threshold=threshold,
    )

    return precision, recall, f1

def evaluate_multi_inter(
    model_path,
    num_rollouts=50,
    num_gibbs_samples=200,
    burnin=100,
    threshold=0.5,
    seed=42,
):
    """Multi-rollout evaluation with separate main effect and interaction metrics."""
    import numpy as np
    import io, sys as _sys

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
    design_network_type = saved_params.get("design_network_type", "dad")

    if sigma2_beta0 is None:
        sigma2_beta0 = (sigma2_alpha0 + 1.0) * (noise_scale ** 2)

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

    interaction_pairs = screening_model.interaction_pairs
    p = num_factors

    print("=" * 64)
    print("Multi-rollout Gibbs evaluation (main effects + interactions)")
    print("=" * 64)
    print(f"Checkpoint:    {model_path}")
    print(f"Policy type:   {design_network_type}")
    print(f"Encoder:       {encoder_type}")
    print(f"Dims:          p={p}, T={T}, interactions={len(interaction_pairs)}")
    print(f"Prior:         pi={expected_active}, noise={noise_scale}")
    print(f"Rollouts:      {num_rollouts}")
    print(f"Gibbs:         {num_gibbs_samples} samples + {burnin} burnin")
    print("-" * 64)

    # Main effect metrics
    main_f1_list, main_tpr_list, main_fdr_list, main_exact_list = [], [], [], []
    # Interaction metrics
    inter_f1_list, inter_tpr_list, inter_fdr_list = [], [], []

    for r in range(num_rollouts):
        if r % 10 == 0:
            print(f"  Rollout {r+1}/{num_rollouts}...")

        # Sample ground truth
        true_z = dist.Bernoulli(
            expected_active * torch.ones(p)
        ).sample()

        scale_main = torch.where(
            true_z.bool(), torch.tensor(scale_active), torch.tensor(scale_inactive)
        )
        true_theta_main = dist.Normal(torch.zeros(p), scale_main).sample()

        interact_active = torch.stack([
            true_z[i] * true_z[j] for i, j in interaction_pairs
        ], dim=-1)
        scale_interact = torch.where(
            interact_active.bool(),
            torch.tensor(scale_active), torch.tensor(scale_inactive)
        )
        true_theta_interact = dist.Normal(
            torch.zeros(len(interaction_pairs)), scale_interact
        ).sample()

        # Run policy
        with torch.no_grad():
            designs, observations = run_policy(
                screening_model, true_z, true_theta_main, true_theta_interact
            )

        # Gibbs sampler (quiet)
        old_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        z_samples, theta_samples, sigma2_samples = gibbs_sampler(
            designs, observations, screening_model,
            num_samples=num_gibbs_samples,
            burnin=burnin,
        )
        _sys.stdout = old_stdout

        # --- Main effect metrics ---
        activity_probs = z_samples.mean(0)
        pred_main = activity_probs > threshold
        true_main = true_z.bool()

        tp = (pred_main & true_main).sum().item()
        fp = (pred_main & ~true_main).sum().item()
        fn = (~pred_main & true_main).sum().item()

        main_tpr = tp / max(tp + fn, 1)
        main_fdr = fp / max(tp + fp, 1)
        if (tp + fp + fn) > 0:
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            main_f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        else:
            main_f1 = 1.0 if true_main.sum() == 0 else 0.0
        main_exact = float(pred_main.equal(true_main))

        main_f1_list.append(main_f1)
        main_tpr_list.append(main_tpr)
        main_fdr_list.append(main_fdr)
        main_exact_list.append(main_exact)

        # --- Interaction metrics ---
        # Predicted interactions from predicted main effects (strong heredity)
        pred_interact = torch.stack([
            pred_main[i] & pred_main[j] for i, j in interaction_pairs
        ])
        true_interact = interact_active.bool()

        itp = (pred_interact & true_interact).sum().item()
        ifp = (pred_interact & ~true_interact).sum().item()
        ifn = (~pred_interact & true_interact).sum().item()

        inter_tpr_val = itp / max(itp + ifn, 1)
        inter_fdr_val = ifp / max(itp + ifp, 1)
        if (itp + ifp + ifn) > 0:
            iprec = itp / max(itp + ifp, 1)
            irec = itp / max(itp + ifn, 1)
            inter_f1_val = 2 * iprec * irec / max(iprec + irec, 1e-8)
        else:
            inter_f1_val = 1.0 if true_interact.sum() == 0 else 0.0

        inter_f1_list.append(inter_f1_val)
        inter_tpr_list.append(inter_tpr_val)
        inter_fdr_list.append(inter_fdr_val)

    # Aggregate
    def fmt(arr):
        a = np.array(arr)
        return f"{a.mean():.4f} +/- {a.std()/np.sqrt(len(a)):.4f}"

    print()
    print("Main Effects:")
    print(f"  F1:          {fmt(main_f1_list)}")
    print(f"  TPR:         {fmt(main_tpr_list)}")
    print(f"  FDR:         {fmt(main_fdr_list)}")
    print(f"  Exact Match: {fmt(main_exact_list)}")
    print()
    print("Interactions:")
    print(f"  F1:          {fmt(inter_f1_list)}")
    print(f"  TPR:         {fmt(inter_tpr_list)}")
    print(f"  FDR:         {fmt(inter_fdr_list)}")
    print()
    print(f"Rollouts:      {num_rollouts}")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained DAD screening policy with interactions"
    )
    parser.add_argument("--model-path",         required=True,  type=str)
    parser.add_argument("--num-rollouts",       default=1,      type=int,
                        help="Use 1 for detailed single-run, 10+ for aggregated metrics")
    parser.add_argument("--num-gibbs-samples",  default=2000,   type=int)
    parser.add_argument("--burnin",             default=500,    type=int)
    parser.add_argument("--threshold",          default=0.5,    type=float)
    parser.add_argument("--seed",               default=42,     type=int)

    # Legacy single-run args
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
        evaluate_multi_inter(
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