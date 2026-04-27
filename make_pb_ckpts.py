import os
import numpy as np
import torch

OUT = "results_pb"
os.makedirs(OUT, exist_ok=True)

PB_GENERATORS = {
    12: [+1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1],
    16: [+1, +1, +1, +1, -1, +1, -1, +1, +1, -1, -1, +1, -1, -1, -1],
}

def pb_matrix(T):
    g = np.array(PB_GENERATORS[T])
    n_cols = T - 1
    M = np.zeros((T, n_cols), dtype=np.float32)
    for i in range(T - 1):
        M[i] = np.roll(g, i)
    M[T - 1] = -1
    return M

configs = [
    (15, 16, 0.2, 0.50, "pb_p15_T16"),
    (10, 16, 0.2, 0.50, "pb_p10_T16"),
    (8,  12, 0.5, 0.25, "pb_p8_T12"),
    (8,  16, 0.5, 0.25, "pb_p8_T16"),
]

for (p, T, pi, noise, label) in configs:
    M = pb_matrix(T)
    assert M.shape[1] >= p, f"PB-{T} has only {M.shape[1]} columns, need p={p}"
    X = M[:, :p]

    state_dict = {}
    state_dict["design_net.prototype"] = torch.zeros(p)
    for t in range(T):
        state_dict[f"design_net.designs.{t}"] = torch.tensor(X[t], dtype=torch.float32)

    params = {
        "seed": 1,
        "num_factors": p,
        "T": T,
        "noise_scale": noise,
        "sigma2_alpha0": 3.0,
        "sigma2_beta0": (3.0 + 1.0) * (noise ** 2),
        "expected_active_proportion": pi,
        "pi_beta_a": 0.0,
        "pi_beta_b": 0.0,
        "scale_active": 1.0,
        "scale_inactive": 0.01,
        "design_network_type": "static",
        "encoder_type": "original",
        "hidden_dim": 64,
        "encoding_dim": 32,
        "num_outer_samples": 0,
        "num_inner_samples": 0,
        "num_steps": 0,
        "lr": 0.0,
        "gamma": 1.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 0.0,
        "diversity_penalty": 0.0,
        "logit_l2_penalty": 0.0,
        "dense_rewards": False,
        "tau_start": 1.0,
        "tau_end": 1.0,
        "tau_anneal_frac": 1.0,
    }

    ckpt = {
        "model_state_dict": state_dict,
        "params": params,
        "loss_history": [],
    }

    path = os.path.join(OUT, f"model_{label}.pt")
    torch.save(ckpt, path)
    print(f"Saved PB checkpoint: {path}  (p={p}, T={T}, {T} rows x {p} cols)")

print()
print("Next step:")
print("  for ckpt in results_pb/model_pb_*.pt; do")
print("    base=$(basename $ckpt .pt)")
print("    python3 evaluate.py --model-path $ckpt \\")
print("      --num-rollouts 50 --seed 42 \\")
print("      --num-gibbs-samples 200 --burnin 100 \\")
print("      > results_pb/${base}_eval.txt 2>&1")
print("    grep -E 'F1:|TPR:' results_pb/${base}_eval.txt | head -2")
print("  done")
