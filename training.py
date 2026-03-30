import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
import pyro
from pyro.infer.util import torch_item
from tqdm import trange

from oed.design import OED
from bound import PriorContrastiveEstimation
from modules import (
    build_screening_network,
    BatchDesignBaseline,
    RandomDesignBaseline,
)
from screening_model import ScreeningExperiment


def single_run(
    seed,
    num_steps,
    num_inner_samples,
    num_outer_samples,
    lr,
    gamma,
    num_factors,
    T,
    noise_scale,
    expected_active_proportion,
    scale_active,
    scale_inactive,
    hidden_dim,
    encoding_dim,
    design_network_type,
    device,
    output_dir,
    adam_betas_wd=None,
    pi_beta_a=0.0,
    pi_beta_b=0.0,
    diversity_penalty=0.1,
    encoder_type="original",
    tau_start=1.0,
    tau_end=0.1,
    tau_anneal_frac=0.5,
):
    if adam_betas_wd is None:
        adam_betas_wd = [0.9, 0.999, 0]

    pyro.clear_param_store()
    if seed == -1:
        seed = torch.randint(0, 10000, (1,)).item()
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    print(f"  seed: {seed}")
    *adam_betas, adam_weight_decay = adam_betas_wd

    if design_network_type == "dad":
        design_net = build_screening_network(
            num_factors=num_factors,
            encoding_dim=encoding_dim,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
        ).to(device)

    elif design_network_type == "static":
        design_net = BatchDesignBaseline(
            T=T,
            design_dim=num_factors,
        ).to(device)

    elif design_network_type == "random":
        design_net = RandomDesignBaseline(
            design_dim=num_factors,
        ).to(device)
        num_steps = 0

    else:
        raise ValueError(
            f"design_network_type={design_network_type} not supported. "
            f"Choose from: 'dad', 'static', 'random'."
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
    )

    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": torch.optim.Adam,
            "optim_args": {
                "lr": lr,
                "betas": adam_betas,
                "weight_decay": adam_weight_decay,
            },
            "gamma": gamma,
        }
    )

    interaction_pairs = [
        (i, j)
        for i in range(num_factors)
        for j in range(i + 1, num_factors)
    ]

    pce_loss = PriorContrastiveEstimation(
        num_outer_samples,
        num_inner_samples,
        scale_active=scale_active,
        scale_inactive=scale_inactive,
        noise_scale=noise_scale,
        diversity_penalty=diversity_penalty,
        interaction_pairs=interaction_pairs,
    )

    oed = OED(screening_model.model, scheduler, pce_loss)
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"loss_history_{run_id}.csv")

    params = {
        "seed": seed,
        "num_factors": num_factors,
        "T": T,
        "noise_scale": noise_scale,
        "expected_active_proportion": expected_active_proportion,
        "pi_beta_a": pi_beta_a,
        "pi_beta_b": pi_beta_b,
        "scale_active": scale_active,
        "scale_inactive": scale_inactive,
        "design_network_type": design_network_type,
        "encoder_type": encoder_type,
        "hidden_dim": hidden_dim,
        "encoding_dim": encoding_dim,
        "num_outer_samples": num_outer_samples,
        "num_inner_samples": num_inner_samples,
        "num_steps": num_steps,
        "lr": lr,
        "gamma": gamma,
        "adam_beta1": adam_betas[0],
        "adam_beta2": adam_betas[1],
        "adam_weight_decay": adam_weight_decay,
        "diversity_penalty": diversity_penalty,
        "tau_start": tau_start,
        "tau_end": tau_end,
        "tau_anneal_frac": tau_anneal_frac,
    }

    print("\n" + "=" * 55)
    print("Screening DAD Training")
    print("=" * 55)
    for k, v in params.items():
        print(f"  {k:30s} {v}")
    print("=" * 55 + "\n")

    loss_history = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "eval_loss"])

        num_steps_range = trange(0, num_steps, desc="Loss: 0.000 ")
        for i in num_steps_range:
            if design_network_type == "dad" and num_steps > 0:
                anneal_steps = max(1, int(num_steps * tau_anneal_frac))
                frac = min(i / anneal_steps, 1.0)
                tau = tau_start * (tau_end / tau_start) ** frac
                design_net.tau = tau

            loss = oed.step()
            loss = torch_item(loss)
            loss_history.append(loss)

            if i % 50 == 0:
                num_steps_range.set_description("Loss: {:.3f} ".format(loss))
                loss_eval = oed.evaluate_loss()
                writer.writerow([i, loss, loss_eval])

            if i % 1000 == 0 and i > 0:
                scheduler.step()

    if len(loss_history) == 0:
        loss = torch_item(pce_loss.differentiable_loss(screening_model.model))
        print(f"Random baseline loss: {loss:.4f}")
    else:
        loss_first50 = np.mean(loss_history[:50])
        loss_last50  = np.mean(loss_history[-51:-1])
        loss_diff50  = loss_last50 / loss_first50 - 1

        print(f"\nTraining complete:")
        print(f"  First 50 steps avg loss : {loss_first50:.4f}")
        print(f"  Last  50 steps avg loss : {loss_last50:.4f}")
        print(f"  Relative change         : {loss_diff50*100:.1f}%")
        print(f"  Loss history saved to   : {csv_path}")

    model_path = os.path.join(output_dir, f"model_{run_id}.pt")
    torch.save(
        {
            "model_state_dict": screening_model.state_dict(),
            "params": params,
            "loss_history": loss_history,
        },
        model_path,
    )
    print(f"  Model saved to          : {model_path}")

    return screening_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DAD Screening Experiment Training"
    )

    parser.add_argument("--seed", default=-1, type=int)

    parser.add_argument("--num-factors", default=20, type=int)
    parser.add_argument("--num-experiments", default=10, type=int)
    parser.add_argument("--noise-scale", default=0.5, type=float)
    parser.add_argument("--expected-active", default=0.2, type=float)
    parser.add_argument("--pi-beta-a", default=0.0, type=float,
                        help="Beta hyperprior shape a for pi (0 = fixed pi)")
    parser.add_argument("--pi-beta-b", default=0.0, type=float,
                        help="Beta hyperprior shape b for pi (0 = fixed pi)")
    parser.add_argument("--scale-active", default=1.0, type=float)
    parser.add_argument("--scale-inactive", default=0.01, type=float)

    parser.add_argument("--hidden-dim", default=64, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--design-network-type", default="dad", type=str,
                        choices=["dad", "static", "random"])

    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--num-inner-samples", default=100, type=int)
    parser.add_argument("--num-outer-samples", default=200, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--adam-betas-wd", nargs="+", type=float,
                        default=[0.9, 0.999, 0])
    parser.add_argument("--diversity-penalty", default=0.1, type=float,
                        help="Weight of the diversity penalty in REINFORCE loss")
    parser.add_argument("--encoder-type", default="original", type=str,
                        choices=["original", "suff"],
                        help="original = MLP sum pooling, suff = sufficient stats")
    parser.add_argument("--tau-start", default=1.0, type=float,
                        help="Initial Gumbel-Softmax temperature")
    parser.add_argument("--tau-end", default=0.1, type=float,
                        help="Final Gumbel-Softmax temperature")
    parser.add_argument("--tau-anneal-frac", default=0.5, type=float,
                        help="Fraction of training steps over which to anneal tau")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--output-dir", default="results", type=str)

    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma=args.gamma,
        num_factors=args.num_factors,
        T=args.num_experiments,
        noise_scale=args.noise_scale,
        expected_active_proportion=args.expected_active,
        pi_beta_a=args.pi_beta_a,
        pi_beta_b=args.pi_beta_b,
        encoder_type=args.encoder_type,
        scale_active=args.scale_active,
        scale_inactive=args.scale_inactive,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        design_network_type=args.design_network_type,
        device=args.device,
        output_dir=args.output_dir,
        adam_betas_wd=args.adam_betas_wd,
        diversity_penalty=args.diversity_penalty,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_anneal_frac=args.tau_anneal_frac,
    )