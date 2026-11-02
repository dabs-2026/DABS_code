import argparse
import json
import os
import time
from datetime import datetime

import torch
import pyro
import numpy as np

from modules import build_screening_network, BatchDesignBaseline
from screening_model import ScreeningExperiment
from evaluate_inter import run_policy, gibbs_sampler


def _stats(flags, widths):
    n = len(flags)
    if n == 0:
        return {"coverage": float("nan"), "coverage_se": float("nan"),
                "mean_width": float("nan"), "n": 0}
    cov = float(np.mean(flags))
    return {
        "coverage": cov,
        "coverage_se": float(np.sqrt(cov * (1.0 - cov) / n)),
        "mean_width": float(np.mean(widths)) if len(widths) > 0 else float("nan"),
        "n": int(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-rollouts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-gibbs-samples", type=int, default=200)
    ap.add_argument("--burnin", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()
    t0 = time.time()

    torch.manual_seed(args.seed); pyro.set_rng_seed(args.seed); np.random.seed(args.seed)

    lo_q = (1.0 - args.level) / 2.0
    hi_q = 1.0 - lo_q

    ck = torch.load(args.model_path, map_location="cpu", weights_only=False)
    p = ck["params"]

    print("=" * 64)
    print("Credible-interval coverage evaluation")
    print("=" * 64)
    print(f"Checkpoint: {args.model_path}")
    print(f"Dims:       p={p['num_factors']}, T={p['T']}")
    print(f"Level:      {args.level:.2f}  (quantiles {lo_q:.4f} / {hi_q:.4f})")
    print(f"Rollouts:   {args.num_rollouts}")
    print("-" * 64)

    dnt = p.get("design_network_type", "dad")
    if dnt == "dad":
        design_net = build_screening_network(
            num_factors  = p["num_factors"],
            encoding_dim = p.get("encoding_dim", 16),
            hidden_dim   = p.get("hidden_dim", 64),
            encoder_type = p.get("encoder_type", "original"),
            mlp_depth    = p.get("mlp_depth", 1),
            pe_dim       = p.get("pe_dim", 4),
            proj_hidden  = p.get("proj_hidden", 64),
        )
    elif dnt == "static":
        design_net = BatchDesignBaseline(T=p["T"], design_dim=p["num_factors"])
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

    P     = screening_model.num_factors
    pairs = screening_model.interaction_pairs

    m_act_f, m_act_w = [], []
    m_ina_f, m_ina_w = [], []
    m_found_f, m_missed_f = [], []
    i_act_f, i_act_w = [], []
    i_ina_f, i_ina_w = [], []

    for r in range(args.num_rollouts):
        if r % 10 == 0:
            print(f"  Rollout {r+1}/{args.num_rollouts}...")
        with torch.no_grad():
            trace = pyro.poutine.trace(screening_model.model).get_trace()
        true_z      = trace.nodes["z"]["value"].detach().cpu().reshape(-1)
        true_main   = trace.nodes["theta_main"]["value"].detach().cpu().reshape(-1)
        true_inter  = trace.nodes["theta_interact"]["value"].detach().cpu().reshape(-1)
        true_sigma2 = (1.0 / trace.nodes["sigma2_inv"]["value"].clamp(min=1e-8)).detach().cpu()
        true_z_int  = (trace.nodes["z_interact"]["value"].detach().cpu().reshape(-1)
                       if "z_interact" in trace.nodes else None)

        designs, observations = run_policy(
            screening_model, true_z, true_main, true_inter,
            true_sigma2=true_sigma2, true_z_interact=true_z_int,
        )
        designs      = designs.detach().cpu()
        observations = observations.detach().cpu()

        z_samples, theta_samples, _, _ = gibbs_sampler(
            designs, observations, screening_model,
            num_samples=args.num_gibbs_samples, burnin=args.burnin,
        )
        z_post = z_samples.float().mean(0)
        th     = theta_samples.float()

        lo = torch.quantile(th, lo_q, dim=0)
        hi = torch.quantile(th, hi_q, dim=0)

        true_theta = torch.cat([true_main, true_inter])
        covered    = (true_theta >= lo) & (true_theta <= hi)
        width      = hi - lo

        cm, wm = covered[:P], width[:P]
        act    = true_z.bool()
        found  = z_post >= args.threshold
        m_act_f.extend(cm[act].tolist());   m_act_w.extend(wm[act].tolist())
        m_ina_f.extend(cm[~act].tolist());  m_ina_w.extend(wm[~act].tolist())
        m_found_f.extend(cm[act & found].tolist())
        m_missed_f.extend(cm[act & ~found].tolist())

        if len(pairs) > 0:
            ci, wi = covered[P:], width[P:]
            true_int_z = (true_z_int if true_z_int is not None
                          else torch.stack([true_z[i] * true_z[j] for i, j in pairs]))
            ia = true_int_z.bool()
            i_act_f.extend(ci[ia].tolist());   i_act_w.extend(wi[ia].tolist())
            i_ina_f.extend(ci[~ia].tolist());  i_ina_w.extend(wi[~ia].tolist())

    res = {
        "main_active":           _stats(m_act_f, m_act_w),
        "main_inactive":         _stats(m_ina_f, m_ina_w),
        "main_active_recovered": _stats(m_found_f, []),
        "main_active_missed":    _stats(m_missed_f, []),
        "int_active":            _stats(i_act_f, i_act_w),
        "int_inactive":          _stats(i_ina_f, i_ina_w),
    }

    print()
    print(f"{'group':<24}{'coverage':>11}{'se':>9}{'width':>11}{'n':>9}")
    print("-" * 64)
    for k, v in res.items():
        w  = v["mean_width"]
        ws = f"{w:.4f}" if w == w else "-"
        se = v["coverage_se"]
        ss = f"{se:.4f}" if se == se else "-"
        cv = v["coverage"]
        cs = f"{cv:.4f}" if cv == cv else "-"
        print(f"{k:<24}{cs:>11}{ss:>9}{ws:>11}{v['n']:>9}")
    print("=" * 64)

    if args.save_json:
        out = {
            "model_path": os.path.abspath(args.model_path),
            "training_params": p,
            "eval_settings": {
                "num_rollouts": args.num_rollouts,
                "seed": args.seed,
                "num_gibbs_samples": args.num_gibbs_samples,
                "burnin": args.burnin,
                "threshold": args.threshold,
                "level": args.level,
            },
            "coverage": res,
            "eval_wallclock_s": time.time() - t0,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".",
                    exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Coverage JSON saved to: {args.save_json}")


if __name__ == "__main__":
    main()
