import argparse
import json
import os
import time
from datetime import datetime
import copy

import numpy as np
import pyro
import torch

from modules import build_screening_network, BatchDesignBaseline
try:
    from modules import RandomDesignBaseline
except ImportError:
    RandomDesignBaseline = None
from screening_model import ScreeningExperiment
from evaluate_inter import run_policy, gibbs_sampler


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
    pred_bin = pred_bin.bool(); true_bin = true_bin.bool()
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
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-rollouts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-gibbs-samples", type=int, default=200)
    ap.add_argument("--burnin", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--eval-expected-active", type=float, default=None,
                    help="If set, override the sparsity used to sample ground truth "
                         "with a fixed Bernoulli prior. Disables Beta hyperprior for truth sampling.")
    ap.add_argument("--eval-pi-beta-a", type=float, default=None)
    ap.add_argument("--eval-pi-beta-b", type=float, default=None)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed); pyro.set_rng_seed(args.seed); np.random.seed(args.seed)

    ck = torch.load(args.model_path, map_location="cpu", weights_only=False)
    p_train = ck["params"]

    print("=" * 64)
    print("Prior-mismatch evaluation")
    print("=" * 64)
    print(f"Checkpoint: {args.model_path}")
    print(f"Train prior: Beta({p_train.get('pi_beta_a')},{p_train.get('pi_beta_b')})"
          f" / expected_active={p_train.get('expected_active_proportion')}")
    if args.eval_expected_active is not None:
        print(f"Eval prior:  fixed pi = {args.eval_expected_active}")
    elif args.eval_pi_beta_a is not None:
        print(f"Eval prior:  Beta({args.eval_pi_beta_a},{args.eval_pi_beta_b})")
    else:
        print("Eval prior:  same as train (no mismatch)")
    print("-" * 64)

    dnt = p_train.get("design_network_type", "dad")
    if dnt == "dad":
        design_net = build_screening_network(
            num_factors  = p_train["num_factors"],
            encoding_dim = p_train.get("encoding_dim", 16),
            hidden_dim   = p_train.get("hidden_dim", 64),
            encoder_type = p_train.get("encoder_type", "original"),
            mlp_depth    = p_train.get("mlp_depth", 1),
            pe_dim       = p_train.get("pe_dim", 4),
            proj_hidden  = p_train.get("proj_hidden", 64),
        )
    elif dnt == "static":
        design_net = BatchDesignBaseline(T=p_train["T"], design_dim=p_train["num_factors"])
    else:
        raise ValueError(f"unknown design_network_type={dnt}")
    sd = ck["model_state_dict"]
    if any(k.startswith("design_net.") for k in sd):
        sd = {k[len("design_net."):]: v for k, v in sd.items() if k.startswith("design_net.")}
    design_net.load_state_dict(sd, strict=False)
    design_net.eval()

    common = dict(
        design_net          = design_net,
        num_factors         = p_train["num_factors"],
        T                   = p_train["T"],
        pi_beta_a           = p_train.get("pi_beta_a", 0.0),
        pi_beta_b           = p_train.get("pi_beta_b", 0.0),
        scale_active        = p_train.get("scale_active", 1.0),
        scale_inactive      = p_train.get("scale_inactive", 0.01),
        noise_scale         = p_train.get("noise_scale", 0.5),
        sigma2_alpha0       = p_train.get("sigma2_alpha0", 3.0),
        sigma2_beta0        = p_train.get("sigma2_beta0", None),
        use_g_prior         = p_train.get("use_g_prior", False),
        strong_heredity     = p_train.get("strong_heredity", False),
        pi_interaction      = p_train.get("pi_interaction", 0.5),
    )
    infer_model = ScreeningExperiment(
        expected_active_proportion = p_train.get("expected_active_proportion",
                                                 p_train.get("expected_active", 0.2)),
        **common,
    )

    truth_common = copy.deepcopy(common)
    if args.eval_expected_active is not None:
        truth_common["pi_beta_a"] = 0.0
        truth_common["pi_beta_b"] = 0.0
        truth_active = args.eval_expected_active
    elif args.eval_pi_beta_a is not None:
        truth_common["pi_beta_a"] = args.eval_pi_beta_a
        truth_common["pi_beta_b"] = args.eval_pi_beta_b
        truth_active = p_train.get("expected_active_proportion", 0.2)
    else:
        truth_active = p_train.get("expected_active_proportion", 0.2)
    truth_model = ScreeningExperiment(
        expected_active_proportion = truth_active,
        **truth_common,
    )

    main_f1s = []; main_tprs = []; main_fdrs = []
    int_f1s = []; int_tprs = []; int_fdrs = []
    z_post_list = []; z_true_list = []
    pairs = infer_model.interaction_pairs

    for r in range(args.num_rollouts):
        if r % 10 == 0:
            print(f"  Rollout {r+1}/{args.num_rollouts}...", flush=True)
        with torch.no_grad():
            trace = pyro.poutine.trace(truth_model.model).get_trace()
        true_z      = trace.nodes["z"]["value"].detach().cpu()
        true_main   = trace.nodes["theta_main"]["value"].detach().cpu()
        true_inter  = trace.nodes["theta_interact"]["value"].detach().cpu()
        true_sigma2 = (1.0 / trace.nodes["sigma2_inv"]["value"].clamp(min=1e-8)).detach().cpu()
        true_z_int  = (trace.nodes["z_interact"]["value"].detach().cpu()
                       if "z_interact" in trace.nodes else None)
        designs, observations = run_policy(
            infer_model, true_z, true_main, true_inter,
            true_sigma2=true_sigma2, true_z_interact=true_z_int,
        )
        designs = designs.detach().cpu(); observations = observations.detach().cpu()

        z_samples, _, _, z_int_samples = gibbs_sampler(
            designs, observations, infer_model,
            num_samples=args.num_gibbs_samples, burnin=args.burnin,
        )
        z_post = z_samples.float().mean(0)
        z_post_list.append([float(v) for v in z_post])
        z_true_list.append([int(v) for v in true_z])

        pred_main = (z_post >= args.threshold).float()
        f1_m, tpr_m, fdr_m, _ = metrics(pred_main, true_z)
        main_f1s.append(f1_m); main_tprs.append(tpr_m); main_fdrs.append(fdr_m)

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
        f1_i, tpr_i, fdr_i, _ = metrics(pred_int, true_int_z)
        int_f1s.append(f1_i); int_tprs.append(tpr_i); int_fdrs.append(fdr_i)

    eval_wallclock_s = time.time() - t0
    print(f"\nMain F1={np.mean(main_f1s):.4f}  TPR={np.mean(main_tprs):.4f}  FDR={np.mean(main_fdrs):.4f}")
    print(f"Int  F1={np.mean(int_f1s):.4f}  TPR={np.mean(int_tprs):.4f}  FDR={np.mean(int_fdrs):.4f}")

    if args.save_json:
        out = {
            "model_path": os.path.abspath(args.model_path),
            "training_params": p_train,
            "eval_prior_override": {
                "expected_active": args.eval_expected_active,
                "pi_beta_a": args.eval_pi_beta_a,
                "pi_beta_b": args.eval_pi_beta_b,
            },
            "eval_settings": {
                "num_rollouts": args.num_rollouts,
                "seed": args.seed,
                "num_gibbs_samples": args.num_gibbs_samples,
                "burnin": args.burnin,
                "threshold": args.threshold,
            },
            "summary_main": {"F1": _summary(main_f1s), "TPR": _summary(main_tprs), "FDR": _summary(main_fdrs)},
            "summary_int":  {"F1": _summary(int_f1s),  "TPR": _summary(int_tprs),  "FDR": _summary(int_fdrs)},
            "per_rollout_main": {"F1": main_f1s, "TPR": main_tprs, "FDR": main_fdrs},
            "per_rollout_int":  {"F1": int_f1s,  "TPR": int_tprs,  "FDR": int_fdrs},
            "z_post_per_rollout": z_post_list,
            "z_true_per_rollout": z_true_list,
            "eval_wallclock_s": eval_wallclock_s,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w") as fjs:
            json.dump(out, fjs, indent=2)
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
