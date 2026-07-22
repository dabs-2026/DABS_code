import argparse
import csv
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import pyro
from pyro.infer.util import torch_item
from tqdm import trange

from oed.design import OED
from bound import PriorContrastiveEstimation
from bound_totalmi import PriorContrastiveEstimationTotalMI
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
    sigma2_alpha0=3.0,
    sigma2_beta0=None,
    logit_l2_penalty=0.01,
    dense_rewards=False,
    static_checkpoint=None,
    warm_start_ckpt=None,
    use_total_mi=False,
    use_g_prior=False,
    strong_heredity=False,
    pi_interaction=0.5,
    grad_clip_max_norm=1.0,
    pe_dim=4,
    mlp_depth=1,
    proj_hidden=64,
    log_every=10,
    early_stop=False,
    early_stop_patience=20,
    early_stop_min_delta=0.001,
    early_stop_min_steps_frac=0.5,
):
    if adam_betas_wd is None:
        adam_betas_wd = [0.9, 0.999, 0]

    pyro.clear_param_store()
    device = torch.device(device)
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    print(f"Requested device          : {device}")
    print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
    print(f"Current device            : {device}")
    print(f"Number of GPUs available  : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name                  : {torch.cuda.get_device_name(0)}")
        print(f"Current GPU memory used   : {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    print("=" * 60)
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
            pe_dim=pe_dim,
            mlp_depth=mlp_depth,
            proj_hidden=proj_hidden,
        ).to(device)

        if static_checkpoint is not None:
            design_net.load_static_floor(static_checkpoint)
            print(f"  Static floor loaded from: {static_checkpoint}")

        if warm_start_ckpt is not None:
            design_net.warm_start_from_static(warm_start_ckpt)
            print(f"  Warm-start initialised from: {warm_start_ckpt}")

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
        sigma2_alpha0=sigma2_alpha0,
        sigma2_beta0=sigma2_beta0,
        use_g_prior=use_g_prior,
        strong_heredity=strong_heredity,
        pi_interaction=pi_interaction,
    )
    if use_g_prior:
        print("[g-prior] ScreeningExperiment using beta | sigma^2 ~ N(0, sigma^2 * scale^2).")
    if strong_heredity:
        print(f"[strong-heredity] z_int ~ Bernoulli(pi_int * z_i * z_j), pi_int={pi_interaction}")
    screening_model = screening_model.to(device)
    resolved_sigma2_beta0 = screening_model.sigma2_beta0

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

    bound_cls = (PriorContrastiveEstimationTotalMI
                 if use_total_mi else PriorContrastiveEstimation)
    pce_loss = bound_cls(
        num_outer_samples,
        num_inner_samples,
        scale_active=scale_active,
        scale_inactive=scale_inactive,
        noise_scale=noise_scale,
        sigma2_alpha0=sigma2_alpha0,
        sigma2_beta0=resolved_sigma2_beta0,
        diversity_penalty=diversity_penalty,
        logit_l2_penalty=logit_l2_penalty,
        dense_rewards=dense_rewards,
        interaction_pairs=interaction_pairs,
        strong_heredity=strong_heredity,
        pi_interaction=pi_interaction,
    )
    if use_total_mi:
        print(f"[Total-MI] Using PriorContrastiveEstimationTotalMI "
              f"(target I(theta; h_T), Gaussian likelihood).")

    oed = OED(
        screening_model.model, scheduler, pce_loss,
        grad_clip_max_norm=grad_clip_max_norm,
    )
    os.makedirs(output_dir, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"loss_history_{run_id}.csv")

    params = {
        "seed": seed,
        "num_factors": num_factors,
        "T": T,
        "noise_scale": noise_scale,
        "sigma2_alpha0": sigma2_alpha0,
        "sigma2_beta0": resolved_sigma2_beta0,
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
        "logit_l2_penalty": logit_l2_penalty,
        "dense_rewards": dense_rewards,
        "use_total_mi": use_total_mi,
        "use_g_prior": use_g_prior,
        "strong_heredity": strong_heredity,
        "pi_interaction": pi_interaction,
        "tau_start": tau_start,
        "tau_end": tau_end,
        "tau_anneal_frac": tau_anneal_frac,
        "grad_clip_max_norm": grad_clip_max_norm,
        "pe_dim": pe_dim,
        "mlp_depth": mlp_depth,
        "proj_hidden": proj_hidden,
        "early_stop": early_stop,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "early_stop_min_steps_frac": early_stop_min_steps_frac,
    }

    print("\n" + "=" * 55)
    print("Screening DAD Training")
    print("=" * 55)
    for k, v in params.items():
        print(f"  {k:30s} {v}")
    print("=" * 55 + "\n")

    loss_history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_t0 = time.time()
    diverged = False
    last_eig_eval = float("nan")
    last_eig_train = float("nan")
    last_tau = tau_start

    import copy
    best_eig_eval = -float("inf")
    best_step = 0
    best_state_dict = None
    no_improve_count = 0
    early_stopped = False
    early_stop_min_steps = int(num_steps * early_stop_min_steps_frac)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "eval_loss", "eig_train", "eig_eval", "tau"])

        num_steps_range = trange(0, num_steps, desc="Loss: 0.000 ")
        tau = tau_start
        for i in num_steps_range:
            if design_network_type == "dad" and num_steps > 0:
                anneal_steps = max(1, int(num_steps * tau_anneal_frac))
                frac         = min(i / anneal_steps, 1.0)
                tau          = tau_start * (tau_end / tau_start) ** frac
                design_net.tau = tau

            loss = oed.step()
            loss = torch_item(loss)
            loss_history.append(loss)
            if not math.isfinite(loss):
                diverged = True

            if i % log_every == 0 or i == num_steps - 1:
                eig_train  = -loss
                loss_eval = oed.evaluate_loss()
                eig_eval   = -loss_eval
                last_eig_train = eig_train
                last_eig_eval  = eig_eval
                last_tau       = tau
                num_steps_range.set_description("EIG: {:.3f} τ: {:.3f}".format(eig_eval, tau))
                writer.writerow([i, loss, loss_eval, eig_train, eig_eval, tau])
                f.flush()

                anneal_steps_local = max(1, int(num_steps * tau_anneal_frac))
                if early_stop and i >= anneal_steps_local:
                    if eig_eval > best_eig_eval + early_stop_min_delta:
                        best_eig_eval = eig_eval
                        best_step     = i
                        best_state_dict = copy.deepcopy(screening_model.state_dict())
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                if early_stop and i >= early_stop_min_steps and \
                   no_improve_count >= early_stop_patience:
                    print(f"\n[early stop] step={i}/{num_steps} "
                          f"best EIG={best_eig_eval:.4f}@step{best_step} "
                          f"(no improvement for {no_improve_count} eval points)")
                    early_stopped = True
                    break

            if i % 1000 == 0 and i > 0:
                scheduler.step()

    if best_state_dict is not None:
        screening_model.load_state_dict(best_state_dict)
        print(f"[early stop] restored best model from step {best_step} "
              f"(EIG_eval={best_eig_eval:.4f})")

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

    train_wallclock_s = time.time() - train_t0
    if torch.cuda.is_available():
        gpu_mem_peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_mem_peak_mb = 0.0
        gpu_name = "cpu"

    summary_path = os.path.join(output_dir, f"training_summary_{run_id}.json")
    summary = {
        "run_id": run_id,
        "model_file": os.path.basename(model_path),
        "loss_history_file": os.path.basename(csv_path),
        "params": params,
        "training": {
            "wallclock_s": train_wallclock_s,
            "n_steps_completed": len(loss_history),
            "n_steps_requested": num_steps,
            "eig_train_final": last_eig_train,
            "eig_eval_final": last_eig_eval,
            "tau_final": last_tau,
            "diverged": bool(diverged),
            "early_stopped": bool(early_stopped),
            "best_eig_eval": (None if best_state_dict is None
                              else float(best_eig_eval)),
            "best_step":     int(best_step),
        },
        "system": {
            "gpu_name": gpu_name,
            "gpu_mem_peak_MB": gpu_mem_peak_mb,
            "device": str(device),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(summary_path, "w") as fjs:
        json.dump(summary, fjs, indent=2)
    print(f"  Training summary saved   : {summary_path}")
    print(f"  Wall-clock              : {train_wallclock_s:.1f} s")
    print(f"  GPU mem peak            : {gpu_mem_peak_mb:.1f} MB")

    print("\n--- Encoding diagnostic (3 example runs) ---")
    try:
        screening_model.eval_policy(n_trace=3, verbose=True)
    except TypeError as _diag_err:
        print(f"  (diagnostic skipped: {_diag_err})")
    return screening_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DAD Screening Experiment Training"
    )

    parser.add_argument("--seed",               default=-1,     type=int)
    parser.add_argument("--num-factors",         default=20,     type=int)
    parser.add_argument("--num-experiments",     default=10,     type=int)
    parser.add_argument("--noise-scale",         default=0.5,    type=float)
    parser.add_argument("--sigma2-alpha0",       default=3.0,    type=float)
    parser.add_argument("--sigma2-beta0",        default=None,   type=float)
    parser.add_argument("--expected-active",     default=0.2,    type=float)
    parser.add_argument("--pi-beta-a",           default=0.0,    type=float)
    parser.add_argument("--pi-beta-b",           default=0.0,    type=float)
    parser.add_argument("--scale-active",        default=1.0,    type=float)
    parser.add_argument("--scale-inactive",      default=0.01,   type=float)

    parser.add_argument("--hidden-dim",          default=64,     type=int)
    parser.add_argument("--encoding-dim",        default=16,     type=int)
    parser.add_argument("--design-network-type", default="dad",  type=str,
                        choices=["dad", "static", "random"])

    parser.add_argument("--num-steps",           default=5000,   type=int)
    parser.add_argument("--num-inner-samples",   default=100,    type=int)
    parser.add_argument("--num-outer-samples",   default=200,    type=int)
    parser.add_argument("--lr",                  default=0.0001, type=float)
    parser.add_argument("--gamma",               default=0.95,   type=float)
    parser.add_argument("--adam-betas-wd",       nargs="+",      type=float,
                        default=[0.9, 0.999, 0])
    parser.add_argument("--diversity-penalty",   default=0.1,    type=float)
    parser.add_argument("--logit-l2-penalty",   default=0.01,   type=float)
    parser.add_argument("--dense-rewards",      action="store_true")
    parser.add_argument("--encoder-type",        default="original", type=str,
                        choices=["original", "suff", "canonical"])
    parser.add_argument("--tau-start",           default=1.0,    type=float)
    parser.add_argument("--tau-end",             default=0.1,    type=float)
    parser.add_argument("--tau-anneal-frac",     default=0.5,    type=float)
    parser.add_argument("--static-checkpoint",   default=None,   type=str)
    parser.add_argument("--warm-start-ckpt",     default=None,   type=str)
    parser.add_argument("--use-total-mi",        action="store_true")
    parser.add_argument("--use-g-prior",           action="store_true")
    parser.add_argument("--strong-heredity",       action="store_true",
                        help="Sample z_int ~ Bernoulli(pi_int * z_i * z_j) "
                             "(non-deterministic interactions, true strong heredity).")
    parser.add_argument("--pi-interaction",        default=0.5, type=float,
                        help="Bernoulli prob for interaction activity within "
                             "the heredity-allowed set (only used if --strong-heredity).")
    parser.add_argument("--grad-clip-max-norm",   default=1.0, type=float,
                        help="Max gradient norm for clip_grad_norm_ (inf-norm). "
                             "Set to 0 or a negative value to disable clipping.")
    parser.add_argument("--pe-dim",               default=4,   type=int,
                        help="Number of sinusoidal positional-encoding frequencies "
                             "(only affects --encoder-type original; ignored by "
                             "suff and canonical encoders).")
    parser.add_argument("--mlp-depth",            default=1,   type=int,
                        help="Number of hidden layers in encoder and emitter MLPs.")
    parser.add_argument("--proj-hidden",          default=64,  type=int,
                        help="Hidden width of the suff-encoder projection MLP.")
    parser.add_argument("--log-every",            default=10,  type=int,
                        help="Write a row of loss_history every N training steps. "
                             "Last step is always logged.")
    parser.add_argument("--early-stop",           action="store_true",
                        help="Enable early stopping when EIG_eval plateaus. "
                             "Saves and restores the best checkpoint by EIG_eval.")
    parser.add_argument("--early-stop-patience", default=20, type=int,
                        help="Number of consecutive eval points (each `log-every` "
                             "steps) without improvement before stopping.")
    parser.add_argument("--early-stop-min-delta", default=0.001, type=float,
                        help="Minimum EIG_eval improvement to count as progress.")
    parser.add_argument("--early-stop-min-steps-frac", default=0.5, type=float,
                        help="Don't allow early stop before this fraction of "
                             "num_steps (lets tau annealing finish).")
    parser.add_argument("--device",              default="cpu",  type=str)
    parser.add_argument("--output-dir",          default="results", type=str)

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
        sigma2_alpha0=args.sigma2_alpha0,
        sigma2_beta0=args.sigma2_beta0,
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
        logit_l2_penalty=args.logit_l2_penalty,
        dense_rewards=args.dense_rewards,
        static_checkpoint=args.static_checkpoint,
        warm_start_ckpt=args.warm_start_ckpt,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_anneal_frac=args.tau_anneal_frac,
        use_total_mi=args.use_total_mi,
        use_g_prior=args.use_g_prior,
        strong_heredity=args.strong_heredity,
        pi_interaction=args.pi_interaction,
        grad_clip_max_norm=args.grad_clip_max_norm,
        pe_dim=args.pe_dim,
        mlp_depth=args.mlp_depth,
        proj_hidden=args.proj_hidden,
        log_every=args.log_every,
        early_stop=args.early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_steps_frac=args.early_stop_min_steps_frac,
    )