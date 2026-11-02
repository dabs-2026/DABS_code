# DABS: Deep Adaptive Bayesian Screening

Reference implementation of Deep Adaptive Bayesian Screening (DABS), a
policy-based adaptive Bayesian experimental design method for discrete
factorial screening under a spike-and-slab prior with strong heredity.

## Code

### Model and objective

- `oed/primitives.py`: labels each random variable as latent parameter, experimental design or observation
- `oed/design.py`: connects the model, loss function and optimizer into a single training step
- `screening_model.py`: spike-and-slab prior with heredity constraint, linear forward map and sequential experiment loop
- `bound.py`: sPCE lower bound and closed-form Student-t marginal likelihood
- `bound_totalmi.py`: total mutual information variant I(theta; h_T), used for the training-objective ablation
- `modules.py`: policy network with sufficient-statistics encoder, set encoder, emitter and static design baseline

### Training and evaluation

- `training.py`: runs from the terminal with configurable arguments, saves the model and loss history to the output directory
- `evaluate_inter.py`: policy rollout and posterior inference over main effects and interactions, with a Metropolis-Hastings step for z and Gibbs steps for z_int, rho, sigma^2 and beta
- `run_eval.py`: evaluates a trained checkpoint and writes a structured eval_summary.json
- `run_eval_prior_mismatch.py`: same evaluation, with ground truth drawn from a different prior than training
- `run_coverage.py`: empirical coverage of the credible intervals at four nominal levels

### Baselines

- `myopic_evaluate.py`: greedy one-step expected information gain rollout
- `run_eval_adaptive_baseline.py`: myopic baseline, wraps myopic_evaluate.py
- `run_eval_pb.py`: Plackett-Burman orthogonal fractional-factorial baseline
- `run_eval_distilled_sensing.py`: multi-stage adaptive baseline with thresholding between stages

### Validation

- `test_gibbs_enumeration.py`: compares the sampler against exact enumeration at p = 4, over the 113 configurations admissible under strong heredity
- `compare_zk_update.py`: total variation distance from the exact posterior, for the Metropolis-Hastings z-update and for the single-site update

## Posterior sampler

The chain alternates between updating z, z_int, rho, sigma^2 and beta, in that
order. The z-update is a Metropolis-Hastings move that proposes a factor
indicator together with the interaction indicators it parents. Under strong
heredity a single-site Gibbs step on z cannot switch an active factor off, so
the move is a Metropolis-Hastings step rather than a Gibbs step.

`gibbs_sampler` in `evaluate_inter.py` takes `zk_update`, which is `"mh"` by
default and also accepts `"block"` and `"legacy"`. `run_eval.py` does not
expose it, so it always uses the default. The single-site comparison reported
in the paper is produced by `compare_zk_update.py`, which sets each mode in
turn.

## Launchers

SLURM array jobs. Adjust the `#SBATCH` header for your cluster. These carry the
exact settings used in the paper, so prefer them over the illustrative commands
below.

- `launchers/run_case_studies.sh`: trains and evaluates the static and DABS arms for the three case studies
- `launchers/run_baselines_matched.sh`: evaluates the myopic, Plackett-Burman and multi-stage adaptive baselines, with model settings taken from each case study's own DABS checkpoints
- `launchers/run_ablations.sh`: generates and submits the hyperparameter sensitivity, method ablation and robustness jobs
- `launchers/run_ablations_reeval.sh`: re-evaluates the ablation checkpoints without retraining
- `launchers/run_eval_only.sh`: re-evaluates existing checkpoints without retraining
- `launchers/run_coverage_c1.sh`, `launchers/run_coverage_c2.sh`, `launchers/run_coverage_c3.sh`: credible-interval coverage for the three case studies

## Requirements

```
pip install -r requirements.txt
```

Python 3.9 or newer.

## Train

Case Study 1 uses the set encoder, `--encoder-type original`. Case Studies 2
and 3 use the sufficient-statistics encoder, `--encoder-type suff`.

`--use-g-prior` selects the sigma^2-scaled slab, beta | sigma^2 ~ N(0, sigma^2
* scale^2), which is the prior used throughout the paper.

```
python training.py \
--num-factors 15 \
--num-experiments 16 \
--design-network-type dad \
--encoder-type original \
--num-steps 30000 \
--lr 1e-3 \
--gamma 0.95 \
--encoding-dim 32 \
--hidden-dim 64 \
--num-outer-samples 6000 \
--num-inner-samples 512 \
--noise-scale 0.5 \
--scale-active 2.0 \
--scale-inactive 0.01 \
--pi-beta-a 2 --pi-beta-b 8 \
--strong-heredity \
--use-g-prior \
--logit-l2-penalty 0.01 \
--output-dir results/case1
```

Use `--design-network-type static` for the static sPCE baseline.

## Evaluate

`--num-gibbs-samples` and `--burnin` set the length of the whole sweep, not
only of the Gibbs steps. The name is kept for backward compatibility with
existing commands.

```
python run_eval.py \
--model-path results/case1/model_20260101_120000.pt \
--num-rollouts 100 \
--num-gibbs-samples 200 \
--burnin 100 \
--threshold 0.5 \
--save-json results/case1/eval_summary.json
```

Reports F1, TPR and FDR for main effects and for two-factor interactions,
together with posterior inclusion probabilities and effect-size estimates.

## Baselines

Each takes the same model arguments as `training.py`.

```
python run_eval_pb.py --num-factors 15 --num-experiments 16 \
--noise-scale 0.5 --scale-active 2.0 --scale-inactive 0.01 \
--pi-beta-a 2 --pi-beta-b 8 --strong-heredity --use-g-prior \
--num-rollouts 100 --seed 1 --save-json results/pb/eval_summary.json
```

Replace `run_eval_pb.py` with `run_eval_adaptive_baseline.py` (myopic) or
`run_eval_distilled_sensing.py` (multi-stage adaptive).
