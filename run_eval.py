import argparse
import os
import torch
import pyro
import numpy as np

from modules import build_screening_network, BatchDesignBaseline
try:
    from modules import RandomDesignBaseline
except ImportError:
    RandomDesignBaseline = None
from screening_model import ScreeningExperiment
from evaluate_inter import run_policy, gibbs_sampler


def f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def metrics_from_z(z_post_mean, z_true, threshold=0.5):
    pred = (z_post_mean >= threshold).float()
    tp = ((pred == 1) & (z_true == 1)).sum().item()
    fp = ((pred == 1) & (z_true == 0)).sum().item()
    fn = ((pred == 0) & (z_true == 1)).sum().item()
    exact = float(torch.equal(pred, z_true.float()))
    return tp, fp, fn, exact


def fdr(tp, fp):
    return (fp / (tp + fp)) if (tp + fp) > 0 else 0.0


def tpr(tp, fn):
    return (tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-rollouts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-gibbs-samples", type=int, default=200)
    ap.add_argument("--burnin", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed); pyro.set_rng_seed(args.seed); np.random.seed(args.seed)

    ck = torch.load(args.model_path, map_location="cpu", weights_only=False)
    p = ck["params"]
    print("=" * 64)
    print("Multi-rollout Gibbs evaluation (main effects + interactions)")
    print("=" * 64)
    print(f"Checkpoint:      {args.model_path}")
    print(f"Policy type:     {p.get('design_network_type', '?')}")
    print(f"Encoder:         {p.get('encoder_type', '?')}")
    print(f"Dims:            p={p['num_factors']}, T={p['T']}, "
          f"interactions={p['num_factors']*(p['num_factors']-1)//2}")
    print(f"Prior:           pi={p.get('expected_active_proportion', p.get('expected_active', '?'))}, "
          f"noise={p.get('noise_scale', '?')}")
    print(f"use_g_prior:     {p.get('use_g_prior', False)}")
    print(f"strong_heredity: {p.get('strong_heredity', False)}  "
          f"(pi_int={p.get('pi_interaction', 0.5)})")
    print(f"Rollouts:        {args.num_rollouts}")
    print(f"Gibbs:           {args.num_gibbs_samples} samples + {args.burnin} burnin")
    print("-" * 64)

    dnt = p.get("design_network_type", "dad")
    if dnt == "dad":
        design_net = build_screening_network(
            num_factors  = p["num_factors"],
            encoding_dim = p.get("encoding_dim", 16),
            hidden_dim   = p.get("hidden_dim", 64),
            encoder_type = p.get("encoder_type", "original"),
        )
    elif dnt == "static":
        design_net = BatchDesignBaseline(T=p["T"], design_dim=p["num_factors"])
    elif dnt == "random" and RandomDesignBaseline is not None:
        design_net = RandomDesignBaseline(design_dim=p["num_factors"])
    else:
        raise ValueError(f"unknown design_network_type={dnt}")
    sd = ck["model_state_dict"]
    if any(k.startswith("design_net.") for k in sd):
        sd = {k[len("design_net."):]: v
              for k, v in sd.items() if k.startswith("design_net.")}
    design_net.load_state_dict(sd, strict=False)
    design_net.eval()

    screening_model = ScreeningExperiment(
        design_net          = design_net,
        num_factors         = p["num_factors"],
        T                   = p["T"],
        expected_active_proportion = p.get("expected_active_proportion",
                                           p.get("expected_active", 0.2)),
        pi_beta_a           = p.get("pi_beta_a", 0.0),
        pi_beta_b           = p.get("pi_beta_b", 0.0),
        scale_active        = p.get("scale_active", 1.0),
        scale_inactive      = p.get("scale_inactive", 0.01),
        noise_scale         = p.get("noise_scale", 0.5),
        sigma2_alpha0       = p.get("sigma2_alpha0", 3.0),
        sigma2_beta0        = p.get("sigma2_beta0", None),
        use_g_prior         = p.get("use_g_prior", False),
        strong_heredity     = p.get("strong_heredity", False),
        pi_interaction      = p.get("pi_interaction", 0.5),
    )

    main_f1s, main_tprs, main_fdrs, em_list = [], [], [], []
    int_f1s,  int_tprs,  int_fdrs           = [], [], []
    pairs = screening_model.interaction_pairs

    for r in range(args.num_rollouts):
        if r % 10 == 0:
            print(f"  Rollout {r+1}/{args.num_rollouts}...")
        with torch.no_grad():
            trace = pyro.poutine.trace(screening_model.model).get_trace()
        true_z      = trace.nodes["z"]["value"].detach().cpu()
        true_main   = trace.nodes["theta_main"]["value"].detach().cpu()
        true_inter  = trace.nodes["theta_interact"]["value"].detach().cpu()
        true_sigma2 = (1.0 / trace.nodes["sigma2_inv"]["value"].clamp(min=1e-8)).detach().cpu()
        true_z_int  = (trace.nodes["z_interact"]["value"].detach().cpu()
                       if "z_interact" in trace.nodes else None)

        designs, observations = run_policy(
            screening_model, true_z, true_main, true_inter,
            true_sigma2=true_sigma2, true_z_interact=true_z_int,
        )
        designs      = designs.detach().cpu()
        observations = observations.detach().cpu()

        z_samples, _, _, z_int_samples = gibbs_sampler(
            designs, observations, screening_model,
            num_samples=args.num_gibbs_samples, burnin=args.burnin,
        )
        z_post = z_samples.float().mean(0)

        tp, fp, fn, em = metrics_from_z(z_post, true_z, args.threshold)
        main_f1s.append(f1(tp, fp, fn))
        main_tprs.append(tpr(tp, fn))
        main_fdrs.append(fdr(tp, fp))
        em_list.append(em)

        if z_int_samples is not None:
            z_pred_int = z_int_samples.float().mean(0)
            true_int_z = (true_z_int if true_z_int is not None
                          else torch.stack([true_z[i]*true_z[j] for i,j in pairs]))
        else:
            z_int_implied = (z_samples[:, [i for i,_ in pairs]] *
                             z_samples[:, [j for _,j in pairs]]).float()
            z_pred_int = z_int_implied.mean(0)
            true_int_z = torch.stack([true_z[i]*true_z[j] for i,j in pairs])

        tp_i, fp_i, fn_i, _ = metrics_from_z(z_pred_int, true_int_z, args.threshold)
        int_f1s.append(f1(tp_i, fp_i, fn_i))
        int_tprs.append(tpr(tp_i, fn_i))
        int_fdrs.append(fdr(tp_i, fp_i))

    def ms(v):
        m = float(np.mean(v)); s = float(np.std(v)) / max(1, len(v) ** 0.5)
        return f"{m:.4f} +/- {s:.4f}"

    print()
    print("Main Effects:")
    print(f"  F1:          {ms(main_f1s)}")
    print(f"  TPR:         {ms(main_tprs)}")
    print(f"  FDR:         {ms(main_fdrs)}")
    print(f"  Exact Match: {ms(em_list)}")
    print()
    print("Interactions:")
    print(f"  F1:          {ms(int_f1s)}")
    print(f"  TPR:         {ms(int_tprs)}")
    print(f"  FDR:         {ms(int_fdrs)}")
    print()
    print(f"Rollouts:      {args.num_rollouts}")
    print("=" * 64)


if __name__ == "__main__":
    main()
