import argparse
import time

import numpy as np
import pyro
import torch

from evaluate_inter import run_policy, gibbs_sampler
from modules import build_screening_network, BatchDesignBaseline
from screening_model import ScreeningExperiment


def prf(pred, true):
    tp = float(((pred > 0.5) & (true > 0.5)).sum())
    fp = float(((pred > 0.5) & (true < 0.5)).sum())
    fn = float(((pred < 0.5) & (true > 0.5)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    return f1, tpr, fdr


def build(ck):
    p = ck["params"]
    dnt = p.get("design_network_type", "dad")
    if dnt == "dad":
        net = build_screening_network(
            num_factors=p["num_factors"],
            encoding_dim=p.get("encoding_dim", 16),
            hidden_dim=p.get("hidden_dim", 64),
            encoder_type=p.get("encoder_type", "original"),
            mlp_depth=p.get("mlp_depth", 1),
            pe_dim=p.get("pe_dim", 4),
            proj_hidden=p.get("proj_hidden", 64),
        )
    else:
        net = BatchDesignBaseline(T=p["T"], design_dim=p["num_factors"])
    sd = ck["model_state_dict"]
    if any(k.startswith("design_net.") for k in sd):
        sd = {k[len("design_net."):]: v for k, v in sd.items()
              if k.startswith("design_net.")}
    net.load_state_dict(sd, strict=False)
    net.eval()
    return ScreeningExperiment(
        design_net=net,
        num_factors=p["num_factors"],
        T=p["T"],
        expected_active_proportion=p.get("expected_active_proportion",
                                         p.get("expected_active", 0.2)),
        pi_beta_a=p.get("pi_beta_a", 0.0),
        pi_beta_b=p.get("pi_beta_b", 0.0),
        scale_active=p.get("scale_active", 1.0),
        scale_inactive=p.get("scale_inactive", 0.01),
        noise_scale=p.get("noise_scale", 0.5),
        sigma2_alpha0=p.get("sigma2_alpha0", 3.0),
        sigma2_beta0=p.get("sigma2_beta0", None),
        use_g_prior=p.get("use_g_prior", False),
        strong_heredity=p.get("strong_heredity", False),
        pi_interaction=p.get("pi_interaction", 0.5),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-rollouts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-gibbs-samples", type=int, default=200)
    ap.add_argument("--burnin", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    ck = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model = build(ck)
    P = model.num_factors
    print(f"p={P}  T={model.T}  heredity={model.strong_heredity}  "
          f"pi_int={model.pi_interaction}  rollouts={args.num_rollouts}")

    acc = {"legacy": {"m": [], "i": []}, "mh": {"m": [], "i": []}}
    t0 = time.time()

    for r in range(args.num_rollouts):
        with torch.no_grad():
            trace = pyro.poutine.trace(model.model).get_trace()
        true_z = trace.nodes["z"]["value"].detach().cpu().reshape(-1)
        tm = trace.nodes["theta_main"]["value"].detach().cpu().reshape(-1)
        ti = trace.nodes["theta_interact"]["value"].detach().cpu().reshape(-1)
        s2 = 1.0 / trace.nodes["sigma2_inv"]["value"].detach().cpu().reshape(-1)
        tzi = trace.nodes["z_interact"]["value"].detach().cpu().reshape(-1) \
            if "z_interact" in trace.nodes else None

        designs, obs = run_policy(model, true_z, tm, ti,
                                  true_sigma2=s2, true_z_interact=tzi)
        designs = torch.stack([d.reshape(-1) for d in designs])
        obs = torch.stack([o.reshape(-1) for o in obs]).reshape(-1)

        for name in ("legacy", "mh"):
            torch.manual_seed(args.seed * 1000 + r)
            zs, _, _, zis = gibbs_sampler(
                designs, obs, model,
                num_samples=args.num_gibbs_samples,
                burnin=args.burnin, zk_update=name,
            )
            acc[name]["m"].append(prf(zs.float().mean(0), true_z))
            if zis is not None and tzi is not None:
                acc[name]["i"].append(prf(zis.float().mean(0), tzi))

        if (r + 1) % 10 == 0:
            print(f"  {r+1}/{args.num_rollouts}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 58)
    print(f"{'':10}{'F1':>10}{'TPR':>10}{'FDR':>10}")
    for part, label in (("m", "main effects"), ("i", "interactions")):
        if not acc["legacy"][part]:
            continue
        print(f"\n{label}")
        for name in ("legacy", "mh"):
            a = np.nanmean(np.array(acc[name][part]), axis=0)
            print(f"  {name:<8}{a[0]:>10.4f}{a[1]:>10.4f}{a[2]:>10.4f}")
        d = (np.nanmean(np.array(acc["mh"][part]), axis=0)
             - np.nanmean(np.array(acc["legacy"][part]), axis=0))
        print(f"  {'delta':<8}{d[0]:>+10.4f}{d[1]:>+10.4f}{d[2]:>+10.4f}")
    print("=" * 58)


if __name__ == "__main__":
    main()
