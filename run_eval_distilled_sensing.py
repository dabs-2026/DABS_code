import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pyro
import torch

from modules import BatchDesignBaseline
from screening_model import ScreeningExperiment


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


def distilled_sensing(true_main, true_sigma2, p, T, K, q_per_stage, rng,
                       ridge_lambda=1e-3):
    active_candidates = list(range(p))
    designs_list = []
    obs_list = []

    base = T // K
    rem = T % K
    T_stages = [base + (1 if k < rem else 0) for k in range(K)]

    last_beta_hat = None
    for k in range(K):
        T_k = T_stages[k]
        if T_k == 0:
            continue
        D_k = np.sign(rng.standard_normal((T_k, p)) + 1e-12).astype(np.float32)
        y_k = D_k @ true_main.numpy() + rng.normal(0, np.sqrt(true_sigma2), size=T_k).astype(np.float32)
        designs_list.append(D_k)
        obs_list.append(y_k)

        D_all = np.concatenate(designs_list, axis=0)
        y_all = np.concatenate(obs_list, axis=0)

        X = D_all
        XtX = X.T @ X
        beta_hat = np.linalg.solve(XtX + ridge_lambda * np.eye(p), X.T @ y_all)
        last_beta_hat = beta_hat

        scores = np.abs(beta_hat[active_candidates])
        keep_n = max(1, int(np.ceil(len(active_candidates) * q_per_stage)))
        if keep_n >= len(active_candidates):
            continue
        order = np.argsort(scores)[::-1]
        active_candidates = [active_candidates[i] for i in order[:keep_n]]

    designs = torch.tensor(np.concatenate(designs_list, axis=0), dtype=torch.float32)
    observations = torch.tensor(np.concatenate(obs_list, axis=0), dtype=torch.float32)

    pred_main = torch.zeros(p)
    pred_main[active_candidates] = 1.0
    return pred_main, designs, observations, last_beta_hat


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
    ap.add_argument("--num-stages", type=int, default=3,
                    help="Number of distillation stages K.")
    ap.add_argument("--keep-fraction", type=float, default=0.5,
                    help="Fraction of candidates retained at each stage.")
    ap.add_argument("--ridge-lambda", type=float, default=1e-3,
                    help="Ridge regularisation on the OLS fit.")
    ap.add_argument("--num-rollouts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-json", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed); pyro.set_rng_seed(args.seed); np.random.seed(args.seed)

    p = args.num_factors
    T = args.num_experiments
    sigma2_beta0 = (args.sigma2_beta0 if args.sigma2_beta0 is not None
                    else (args.sigma2_alpha0 + 1.0) * (args.noise_scale ** 2))

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

    print("=" * 64)
    print(f"Distilled Sensing baseline (K={args.num_stages}, keep={args.keep_fraction})")
    print("=" * 64)
    print(f"Dims:           p={p}, T={T}")
    print(f"Prior:          Beta({args.pi_beta_a}, {args.pi_beta_b})")
    print(f"Stages:         {args.num_stages}, keep fraction={args.keep_fraction}")
    print(f"Rollouts:       {args.num_rollouts}   seed={args.seed}")
    print("-" * 64)

    rng = np.random.default_rng(args.seed)

    main_f1s, main_tprs, main_fdrs, main_precs, em_list = [], [], [], [], []
    int_f1s,  int_tprs,  int_fdrs,  int_precs           = [], [], [], []
    z_post_main_list, z_true_main_list = [], []

    for r in range(args.num_rollouts):
        if r % 10 == 0 or r == args.num_rollouts - 1:
            print(f"  Rollout {r+1}/{args.num_rollouts}  ({time.time()-t0:.1f}s)", flush=True)

        with torch.no_grad():
            trace = pyro.poutine.trace(screening_model.model).get_trace()
        true_z      = trace.nodes["z"]["value"].detach().cpu()
        true_main   = trace.nodes["theta_main"]["value"].detach().cpu()
        true_inter  = trace.nodes["theta_interact"]["value"].detach().cpu()
        true_sigma2 = (1.0 / trace.nodes["sigma2_inv"]["value"].clamp(min=1e-8)).detach().cpu().item()
        true_z_int  = (trace.nodes["z_interact"]["value"].detach().cpu()
                       if "z_interact" in trace.nodes else None)

        pred_main, designs, observations, beta_hat = distilled_sensing(
            true_main=true_main,
            true_sigma2=true_sigma2,
            p=p, T=T,
            K=args.num_stages,
            q_per_stage=args.keep_fraction,
            rng=rng,
            ridge_lambda=args.ridge_lambda,
        )

        f1_m, tpr_m, fdr_m, prec_m = metrics(pred_main, true_z)
        main_f1s.append(f1_m); main_tprs.append(tpr_m)
        main_fdrs.append(fdr_m); main_precs.append(prec_m)
        em_list.append(float(torch.equal(pred_main.bool(), true_z.bool())))
        z_post_main_list.append([float(v) for v in pred_main])
        z_true_main_list.append([int(v) for v in true_z])

        if pairs and len(pairs) > 0:
            pred_int_bin = torch.zeros(len(pairs))
            true_int_z = (true_z_int if true_z_int is not None
                          else torch.stack([true_z[i]*true_z[j] for i,j in pairs]))
            f1_i, tpr_i, fdr_i, prec_i = metrics(pred_int_bin, true_int_z)
        else:
            f1_i = tpr_i = fdr_i = prec_i = 0.0
        int_f1s.append(f1_i); int_tprs.append(tpr_i)
        int_fdrs.append(fdr_i); int_precs.append(prec_i)

    eval_wallclock_s = time.time() - t0
    print(f"\n  total eval wallclock: {eval_wallclock_s:.1f}s "
          f"({eval_wallclock_s/max(args.num_rollouts,1):.3f}s per rollout)")
    print(f"\nMain Effects: F1={np.mean(main_f1s):.4f}  TPR={np.mean(main_tprs):.4f}  FDR={np.mean(main_fdrs):.4f}")

    if args.save_json:
        training_params = {
            "arm": "distilled_sensing",
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
            "design_network_type": "distilled_sensing",
            "use_g_prior": args.use_g_prior,
            "strong_heredity": args.strong_heredity,
            "pi_interaction": args.pi_interaction,
            "num_stages": args.num_stages,
            "keep_fraction": args.keep_fraction,
            "ridge_lambda": args.ridge_lambda,
        }
        out = {
            "model_path": None,
            "training_params": training_params,
            "eval_settings": {
                "num_rollouts": args.num_rollouts,
                "seed": args.seed,
                "num_gibbs_samples": 0,
                "burnin": 0,
                "threshold": 0.5,
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
