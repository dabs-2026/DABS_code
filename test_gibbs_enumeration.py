import argparse
import itertools
import math

import torch

from evaluate_inter import (
    build_design_matrix,
    gibbs_sampler,
    _student_log_marginal,
)
from screening_model import ScreeningExperiment


def admissible_configs(p, interaction_pairs):
    configs = []
    for bits in itertools.product([0.0, 1.0], repeat=p):
        z = torch.tensor(bits)
        free = [idx for idx, (i, j) in enumerate(interaction_pairs)
                if z[i] > 0.5 and z[j] > 0.5]
        for sub in itertools.product([0.0, 1.0], repeat=len(free)):
            z_int = torch.zeros(len(interaction_pairs))
            for idx, val in zip(free, sub):
                z_int[idx] = val
            configs.append((z, z_int))
    return configs


def log_prior(z, z_int, interaction_pairs, p, pi_fixed,
              pi_interaction, pi_beta_a, pi_beta_b, use_hyperprior):
    k = int(z.sum().item())
    if use_hyperprior:
        lp = (torch.lgamma(torch.tensor(pi_beta_a + k))
              + torch.lgamma(torch.tensor(pi_beta_b + p - k))
              - torch.lgamma(torch.tensor(pi_beta_a + pi_beta_b + p)))
        lp = lp - (torch.lgamma(torch.tensor(pi_beta_a))
                   + torch.lgamma(torch.tensor(pi_beta_b))
                   - torch.lgamma(torch.tensor(pi_beta_a + pi_beta_b)))
        lp = lp.item()
    else:
        lp = k * math.log(pi_fixed) + (p - k) * math.log(1.0 - pi_fixed)

    for idx, (i, j) in enumerate(interaction_pairs):
        if z[i] > 0.5 and z[j] > 0.5:
            lp += (math.log(pi_interaction) if z_int[idx] > 0.5
                   else math.log(1.0 - pi_interaction))
    return lp


def exact_posterior(y, X, configs, interaction_pairs, model):
    logs = []
    for z, z_int in configs:
        ll = _student_log_marginal(
            y, X, z, interaction_pairs,
            model.scale_active, model.scale_inactive,
            model.sigma2_alpha0, model.sigma2_beta0,
            z_interact=z_int,
        ).item()
        lp = log_prior(
            z, z_int, interaction_pairs, model.num_factors,
            model.pi_fixed, model.pi_interaction,
            model.pi_beta_a, model.pi_beta_b, model.use_pi_hyperprior,
        )
        logs.append(ll + lp)
    logs = torch.tensor(logs)
    return torch.softmax(logs, dim=0)


def reference_gibbs(designs, observations, model, num_samples, burnin, seed=0):
    torch.manual_seed(seed)
    p = model.num_factors
    pairs = model.interaction_pairs
    X = build_design_matrix(designs, pairs)
    y = observations

    pi = model.pi_fixed
    pi_int = model.pi_interaction

    pairs_touching = [[] for _ in range(p)]
    for idx, (i, j) in enumerate(pairs):
        pairs_touching[i].append(idx)
        pairs_touching[j].append(idx)

    def marg(z, z_int):
        return _student_log_marginal(
            y, X, z, pairs,
            model.scale_active, model.scale_inactive,
            model.sigma2_alpha0, model.sigma2_beta0,
            z_interact=z_int,
        )

    z = torch.bernoulli(pi * torch.ones(p))
    ii = torch.tensor([i for i, _ in pairs])
    jj = torch.tensor([j for _, j in pairs])
    z_int = torch.bernoulli(pi_int * (z[ii] * z[jj]))

    draws = []
    for step in range(burnin + num_samples):
        for k in range(p):
            z1 = z.clone(); z1[k] = 1.0
            z0 = z.clone(); z0[k] = 0.0

            free = [idx for idx in pairs_touching[k]
                    if z[pairs[idx][0] if pairs[idx][1] == k else pairs[idx][1]] > 0.5]

            mask_off = torch.zeros_like(z_int)
            for idx in pairs_touching[k]:
                mask_off[idx] = 1.0
            z_int0 = z_int * (1.0 - mask_off)

            log_child = 0.0
            for idx in free:
                log_child += (math.log(pi_int) if z_int[idx] > 0.5
                              else math.log(1.0 - pi_int))

            log_p1 = marg(z1, z_int) + math.log(pi) + log_child
            log_p0 = marg(z0, z_int0) + math.log(1.0 - pi)

            prob = torch.exp(log_p1 - torch.logaddexp(log_p1, log_p0)).clamp(0.0, 1.0)
            z[k] = torch.bernoulli(prob)

        z_int = z_int * (z[ii] * z[jj])
        for idx, (i, j) in enumerate(pairs):
            if z[i] > 0.5 and z[j] > 0.5:
                z1 = z_int.clone(); z1[idx] = 1.0
                z0 = z_int.clone(); z0[idx] = 0.0
                log_p1 = marg(z, z1) + math.log(pi_int)
                log_p0 = marg(z, z0) + math.log(1.0 - pi_int)
                prob = torch.exp(log_p1 - torch.logaddexp(log_p1, log_p0)).clamp(0.0, 1.0)
                z_int[idx] = torch.bernoulli(prob)

        if step >= burnin:
            draws.append((z.clone(), z_int.clone()))
    return draws


def block_gibbs(designs, observations, model, num_samples, burnin, seed=0):
    torch.manual_seed(seed)
    p = model.num_factors
    pairs = model.interaction_pairs
    X = build_design_matrix(designs, pairs)
    y = observations

    pi = model.pi_fixed
    pi_int = model.pi_interaction
    log_pi_int = math.log(pi_int)
    log_1m_pi_int = math.log(1.0 - pi_int)

    pairs_touching = [[] for _ in range(p)]
    for idx, (i, j) in enumerate(pairs):
        pairs_touching[i].append(idx)
        pairs_touching[j].append(idx)
    partner = {}
    for idx, (i, j) in enumerate(pairs):
        partner[(i, idx)] = j
        partner[(j, idx)] = i

    def marg(z, z_int):
        return _student_log_marginal(
            y, X, z, pairs,
            model.scale_active, model.scale_inactive,
            model.sigma2_alpha0, model.sigma2_beta0,
            z_interact=z_int,
        )

    z = torch.bernoulli(pi * torch.ones(p))
    ii = torch.tensor([i for i, _ in pairs])
    jj = torch.tensor([j for _, j in pairs])
    z_int = torch.bernoulli(pi_int * (z[ii] * z[jj]))

    draws = []
    for step in range(burnin + num_samples):
        for k in range(p):
            z1 = z.clone(); z1[k] = 1.0
            z0 = z.clone(); z0[k] = 0.0

            free = [idx for idx in pairs_touching[k]
                    if z[partner[(k, idx)]] > 0.5]

            base = z_int.clone()
            for idx in pairs_touching[k]:
                base[idx] = 0.0

            logs, states = [], []
            for bits in itertools.product([0.0, 1.0], repeat=len(free)):
                cand = base.clone()
                lp = 0.0
                for idx, val in zip(free, bits):
                    cand[idx] = val
                    lp += log_pi_int if val > 0.5 else log_1m_pi_int
                logs.append(marg(z1, cand) + lp)
                states.append(cand)
            log_on = torch.logsumexp(torch.stack(logs), dim=0) + math.log(pi)
            log_off = marg(z0, base) + math.log(1.0 - pi)

            prob = torch.exp(log_on - torch.logaddexp(log_on, log_off)).clamp(0.0, 1.0)
            if torch.bernoulli(prob) > 0.5:
                z[k] = 1.0
                w = torch.softmax(torch.stack(logs), dim=0)
                z_int = states[int(torch.multinomial(w, 1).item())]
            else:
                z[k] = 0.0
                z_int = base

        for idx, (i, j) in enumerate(pairs):
            if z[i] > 0.5 and z[j] > 0.5:
                c1 = z_int.clone(); c1[idx] = 1.0
                c0 = z_int.clone(); c0[idx] = 0.0
                lp1 = marg(z, c1) + log_pi_int
                lp0 = marg(z, c0) + log_1m_pi_int
                pr = torch.exp(lp1 - torch.logaddexp(lp1, lp0)).clamp(0.0, 1.0)
                z_int[idx] = torch.bernoulli(pr)

        if step >= burnin:
            draws.append((z.clone(), z_int.clone()))
    return draws


def empirical(draws, index):
    counts = torch.zeros(len(index))
    for z, z_int in draws:
        key = (tuple(z.tolist()), tuple(z_int.tolist()))
        counts[index[key]] += 1
    return counts / counts.sum()


def tv(a, b):
    return 0.5 * (a - b).abs().sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-gibbs-samples", type=int, default=200000)
    ap.add_argument("--burnin", type=int, default=2000)
    ap.add_argument("--scale-active", type=float, default=2.0)
    ap.add_argument("--scale-inactive", type=float, default=0.01)
    ap.add_argument("--noise-scale", type=float, default=0.5)
    ap.add_argument("--sigma2-alpha0", type=float, default=3.0)
    ap.add_argument("--pi", type=float, default=0.3)
    ap.add_argument("--pi-interaction", type=float, default=0.5)
    ap.add_argument("--pi-beta-a", type=float, default=0.0)
    ap.add_argument("--pi-beta-b", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    model = ScreeningExperiment(
        design_net=None,
        num_factors=args.p,
        T=args.T,
        expected_active_proportion=args.pi,
        pi_beta_a=args.pi_beta_a,
        pi_beta_b=args.pi_beta_b,
        scale_active=args.scale_active,
        scale_inactive=args.scale_inactive,
        noise_scale=args.noise_scale,
        sigma2_alpha0=args.sigma2_alpha0,
        use_g_prior=True,
        strong_heredity=True,
        pi_interaction=args.pi_interaction,
    )
    pairs = model.interaction_pairs

    designs = torch.where(torch.rand(args.T, args.p) > 0.5, 1.0, -1.0)
    X = build_design_matrix(designs, pairs)

    true_z = torch.tensor([1.0, 1.0, 0.0, 0.0])
    true_z_int = torch.zeros(len(pairs))
    true_z_int[0] = 1.0
    sigma = args.noise_scale
    scale = torch.where(
        torch.cat([true_z, true_z_int]).bool(),
        torch.tensor(args.scale_active),
        torch.tensor(args.scale_inactive),
    ) * sigma
    beta = torch.randn(X.shape[1]) * scale
    y = X @ beta + sigma * torch.randn(args.T)

    configs = admissible_configs(args.p, pairs)
    index = {(tuple(z.tolist()), tuple(zi.tolist())): n
             for n, (z, zi) in enumerate(configs)}
    print(f"admissible configurations: {len(configs)}")

    exact = exact_posterior(y, X, configs, pairs, model)

    print("\nrunning shipped sampler ...")
    torch.manual_seed(args.seed + 1)
    z_s, _, _, zint_s = gibbs_sampler(
        designs, y, model,
        num_samples=args.num_gibbs_samples, burnin=args.burnin,
    )
    shipped = empirical(list(zip(z_s, zint_s)), index)

    print("\nrunning legacy sampler (no children's prior at all) ...")
    torch.manual_seed(args.seed + 4)
    z_l, _, _, zint_l = gibbs_sampler(
        designs, y, model,
        num_samples=args.num_gibbs_samples, burnin=args.burnin,
        zk_update="legacy",
    )
    legacy = empirical(list(zip(z_l, zint_l)), index)

    print("\nrunning reference sampler with the children's prior ...")
    ref_draws = reference_gibbs(
        designs, y, model,
        num_samples=args.num_gibbs_samples, burnin=args.burnin,
        seed=args.seed + 2,
    )
    reference = empirical(ref_draws, index)

    print("\nrunning block update marginalising the children ...")
    blk_draws = block_gibbs(
        designs, y, model,
        num_samples=args.num_gibbs_samples, burnin=args.burnin,
        seed=args.seed + 3,
    )
    block = empirical(blk_draws, index)

    half = len(z_s) // 2
    h1 = empirical(list(zip(z_s[:half], zint_s[:half])), index)
    h2 = empirical(list(zip(z_s[half:], zint_s[half:])), index)

    print("\n" + "=" * 70)
    print(f"TV(exact, shipped / MH move)  = {tv(exact, shipped):.5f}")
    print(f"TV(exact, marginalised block) = {tv(exact, block):.5f}")
    print(f"TV(exact, children's prior)   = {tv(exact, reference):.5f}")
    print(f"TV(exact, legacy step)        = {tv(exact, legacy):.5f}")
    print(f"TV(half 1, half 2)            = {tv(h1, h2):.5f}   [Monte Carlo floor]")
    print("=" * 70)

    order = torch.argsort(exact, descending=True)[:10]
    print("\ntop configurations by exact posterior mass")
    print(f"{'z':<8}{'z_int':<10}{'exact':>9}{'shipped':>10}{'children':>10}{'block':>10}")
    for n in order.tolist():
        z, zi = configs[n]
        zs = "".join(str(int(v)) for v in z.tolist())
        zis = "".join(str(int(v)) for v in zi.tolist())
        print(f"{zs:<8}{zis:<10}{exact[n]:>9.4f}{shipped[n]:>10.4f}"
              f"{reference[n]:>10.4f}{block[n]:>10.4f}")


if __name__ == "__main__":
    main()
