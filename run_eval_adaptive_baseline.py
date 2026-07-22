import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from screening_model import ScreeningExperiment
from modules import build_screening_network
from evaluate_inter import gibbs_sampler
from myopic_evaluate import (
    enumerate_designs,
    random_candidates,
    run_myopic_rollout,
)


def _summary(values):
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan"),
                "median": float("nan"), "min": float("nan"), "max": float("nan"),
                "n": 0}
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = s / max(1.0, n ** 0.5)
    return {
        "mean": m, "se": se,
        "ci95_lo": m - 1.96 * se, "ci95_hi": m + 1.96 * se,
        "std": s,
        "median": float(np.median(arr)),
        "min": float(arr.min()), "max": float(arr.max()),
        "n": int(n),
    }


def metrics(pred_bin, true_bin):
    pred_bin = pred_bin.bool()
    true_bin = true_bin.bool()
    tp = (pred_bin & true_bin).sum().item()
    fp = (pred_bin & ~true_bin).sum().item()
    fn = (~pred_bin & true_bin).sum().item()
    denom_f1 = 2 * tp + fp + fn
    f1 = (2 * tp / denom_f1) if denom_f1 > 0 else 0.0
    tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fdr = (fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return f1, tpr, fdr, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-factors", type=int, required=True)
    ap.add_argument("--num-experiments", type=int, required=True)
    ap.add_argument("--noise-scale", type=float, default=0.5)
    ap.add_argument("--expected-active", type=float, default=0.5)
    ap.add_argument("--scale-active", type=float, default=1.0)
    ap.add_argument("--scale-inactive", type=float, default=0.01)
    ap.add_argument("--sigma2-alpha0", type=float, default=3.0)
    ap.add_argument("--sigma2-beta0", type=float, default=None)
    ap.add_argument("--pi-beta-a", type=float, default=4.0)
    ap.add_argument("--pi-beta-b", type=float, default=6.0)
    ap.add_argument("--pi-interaction", type=float, default=0.5)
    ap.add_argument("--strong-heredity", action="store_true")
    ap.add_argument("--use-g-prior", action="store_true")
    ap.add_argument("--n-contrastive", type=int, default=100)
    ap.add_argument("--n-y-samples", type=int, default=3)
    ap.add_argument("--num-candidates", type=int, default=2000,
                    help="Random-candidate count when p > 12.")
    ap.add_argument("--num-rollouts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-gibbs-samples", type=int, default=200)
    ap.add_argument("--burnin", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save-json", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed); pyro.set_rng_seed(args.seed); np.random.seed(args.seed)

    p = args.num_factors
    T = args.num_experiments
    sigma2_beta0 = (args.sigma2_beta0 if args.sigma2_beta0 is not None
                    else (args.sigma2_alpha0 + 1.0) * (args.noise_scale ** 2))

    from modules import BatchDesignBaseline
    dummy_net = BatchDesignBaseline(T=T, design_dim=p)
    screening_model = ScreeningExperiment(
        design_net          = dummy_net,
        num_factors         = p,
        T                   = T,
        expected_active_proportion = args.expected_active,
        pi_beta_a           = args.pi_beta_a,
        pi_beta_b           = args.pi_beta_b,
        scale_active        = args.scale_active,
        scale_inactive      = args.scale_inactive,
        noise_scale         = args.noise_scale,
        sigma2_alpha0       = args.sigma2_alpha0,
        sigma2_beta0        = sigma2_beta0,
        use_g_prior         = args.use_g_prior,
        strong_heredity     = args.strong_heredity,
        pi_interaction      = args.pi_interaction,
    )
    pairs = screening_model.interaction_pairs

    if 2 ** p <= args.num_candidates:
        candidates = enumerate_designs(p, device=args.device)
        cand_mode = f"enumerated {2**p}"
    else:
        candidates = random_candidates(p, n=args.num_candidates, device=args.device)
        cand_mode = f"random {args.num_candidates}"

    pi_for_eig = (args.pi_beta_a / max(args.pi_beta_a + args.pi_beta_b, 1e-8)
                  if args.pi_beta_a > 0 else args.expected_active)

    print("=" * 64)
    print("Policy-free adaptive baseline (greedy one-step EIG over candidates)")
    print("=" * 64)
    print(f"Dims:           p={p}, T={T}")
    print(f"Prior:          Beta({args.pi_beta_a}, {args.pi_beta_b})  pi_eig={pi_for_eig:.3f}")
    print(f"Scales:         active={args.scale_active}, inactive={args.scale_inactive}")
    print(f"Noise:          {args.noise_scale}  sigma2~InvGamma({args.sigma2_alpha0}, {sigma2_beta0:.3f})")
    print(f"Heredity:       strong={args.strong_heredity}  pi_int={args.pi_interaction}")
    print(f"Candidates:     {cand_mode}")
    print(f"EIG estimator:  L={args.n_contrastive}  S={args.n_y_samples}")
    print(f"Rollouts:       {args.num_rollouts}   seed={args.seed}")
    print(f"Gibbs:          {args.num_gibbs_samples} samples + {args.burnin} burnin")
    print("-" * 64)

    main_f1s, main_tprs, main_fdrs, main_precs, em_list = [], [], [], [], []
    int_f1s,  int_tprs,  int_fdrs,  int_precs           = [], [], [], []
    z_post_main_list, z_true_main_list = [], []

    for r in range(args.num_rollouts):
        if r % 5 == 0 or r == args.num_rollouts - 1:
            print(f"  Rollout {r+1}/{args.num_rollouts}  ({time.time()-t0:.1f}s)", flush=True)

        with torch.no_grad():
            trace = pyro.poutine.trace(screening_model.model).get_trace()
        true_z      = trace.nodes["z"]["value"].detach().cpu()
        true_main   = trace.nodes["theta_main"]["value"].detach().cpu()
        true_inter  = trace.nodes["theta_interact"]["value"].detach().cpu()
        true_sigma2 = (1.0 / trace.nodes["sigma2_inv"]["value"].clamp(min=1e-8)).detach().cpu().item()
        true_z_int  = (trace.nodes["z_interact"]["value"].detach().cpu()
                       if "z_interact" in trace.nodes else None)

        designs, observations = run_myopic_rollout(
            true_z, true_main, true_inter, true_sigma2,
            T, p,
            args.scale_active, args.scale_inactive,
            args.sigma2_alpha0, sigma2_beta0,
            pi_for_eig, pairs,
            n_contrastive=args.n_contrastive,
            n_y_samples=args.n_y_samples,
            candidates=candidates,
            device=args.device,
            verbose=False,
        )

        z_samples, _, _, z_int_samples = gibbs_sampler(
            designs, observations, screening_model,
            num_samples=args.num_gibbs_samples, burnin=args.burnin,
        )
        z_post = z_samples.float().mean(0)

        pred_main = (z_post >= args.threshold).float()
        f1_m, tpr_m, fdr_m, prec_m = metrics(pred_main, true_z)
        main_f1s.append(f1_m); main_tprs.append(tpr_m)
        main_fdrs.append(fdr_m); main_precs.append(prec_m)
        em_list.append(float(torch.equal(pred_main.bool(), true_z.bool())))
        z_post_main_list.append([float(v) for v in z_post])
        z_true_main_list.append([int(v) for v in true_z])

        if z_int_samples is not None:
            z_pred_int = z_int_samples.float().mean(0)
            true_int_z = (true_z_int if true_z_int is not None
                          else torch.stack([true_z[i]*true_z[j] for i,j in pairs]))
        else:
            z_int_implied = (z_samples[:, [i for i,_ in pairs]] *
                             z_samples[:, [j for _,j in pairs]]).float()
            z_pred_int = z_int_implied.mean(0)
            true_int_z = torch.stack([true_z[i]*true_z[j] for i,j in pairs])
        pred_int = (z_pred_int >= args.threshold).float()
        f1_i, tpr_i, fdr_i, prec_i = metrics(pred_int, true_int_z)
        int_f1s.append(f1_i); int_tprs.append(tpr_i)
        int_fdrs.append(fdr_i); int_precs.append(prec_i)

    eval_wallclock_s = time.time() - t0
    print(f"\n  total eval wallclock: {eval_wallclock_s:.1f}s "
          f"({eval_wallclock_s/max(args.num_rollouts,1):.2f}s per rollout)")

    print()
    print("Main Effects:")
    print(f"  F1:        {np.mean(main_f1s):.4f} +/- {np.std(main_f1s, ddof=1)/np.sqrt(len(main_f1s)):.4f}")
    print(f"  TPR:       {np.mean(main_tprs):.4f}")
    print(f"  FDR:       {np.mean(main_fdrs):.4f}")
    print("Interactions:")
    print(f"  F1:        {np.mean(int_f1s):.4f} +/- {np.std(int_f1s, ddof=1)/np.sqrt(len(int_f1s)):.4f}")
    print(f"  TPR:       {np.mean(int_tprs):.4f}")
    print(f"  FDR:       {np.mean(int_fdrs):.4f}")

    if args.save_json:
        training_params = {
            "arm": "adaptive_baseline",
            "seed": args.seed,
            "num_factors": p,
            "T": T,
            "noise_scale": args.noise_scale,
            "sigma2_alpha0": args.sigma2_alpha0,
            "sigma2_beta0": sigma2_beta0,
            "expected_active_proportion": args.expected_active,
            "pi_beta_a": args.pi_beta_a,
            "pi_beta_b": args.pi_beta_b,
            "scale_active": args.scale_active,
            "scale_inactive": args.scale_inactive,
            "design_network_type": "adaptive_baseline",
            "use_g_prior": args.use_g_prior,
            "strong_heredity": args.strong_heredity,
            "pi_interaction": args.pi_interaction,
            "n_contrastive": args.n_contrastive,
            "n_y_samples": args.n_y_samples,
            "num_candidates": args.num_candidates,
        }
        out = {
            "model_path": None,
            "training_params": training_params,
            "eval_settings": {
                "num_rollouts": args.num_rollouts,
                "seed": args.seed,
                "num_gibbs_samples": args.num_gibbs_samples,
                "burnin": args.burnin,
                "threshold": args.threshold,
            },
            "per_rollout_main": {
                "F1":          [float(v) for v in main_f1s],
                "TPR":         [float(v) for v in main_tprs],
                "FDR":         [float(v) for v in main_fdrs],
                "precision":   [float(v) for v in main_precs],
                "exact_match": [float(v) for v in em_list],
            },
            "per_rollout_int": {
                "F1":        [float(v) for v in int_f1s],
                "TPR":       [float(v) for v in int_tprs],
                "FDR":       [float(v) for v in int_fdrs],
                "precision": [float(v) for v in int_precs],
            },
            "summary_main": {
                "F1":          _summary(main_f1s),
                "TPR":         _summary(main_tprs),
                "FDR":         _summary(main_fdrs),
                "precision":   _summary(main_precs),
                "exact_match": _summary(em_list),
            },
            "summary_int": {
                "F1":        _summary(int_f1s),
                "TPR":       _summary(int_tprs),
                "FDR":       _summary(int_fdrs),
                "precision": _summary(int_precs),
            },
            "z_post_per_rollout": z_post_main_list,
            "z_true_per_rollout": z_true_main_list,
            "eval_wallclock_s": eval_wallclock_s,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".",
                    exist_ok=True)
        with open(args.save_json, "w") as fjs:
            json.dump(out, fjs, indent=2)
        print(f"  Eval JSON saved to: {args.save_json}")


if __name__ == "__main__":
    main()
