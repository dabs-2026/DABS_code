# DABS: Deep Adaptive Bayesian Screening

Reference implementation of *Deep Adaptive Bayesian Screening (DABS)*, a policy-based
adaptive Bayesian experimental design method for discrete factorial screening under a
spike-and-slab prior with strong heredity.

## Code

**Model and objective**

* `oed/primitives.py`: labels each random variable as latent parameter, experimental design or observation
* `oed/design.py`: connects the model, loss function and optimizer into a single training step
* `screening_model.py`: spike-and-slab prior with heredity constraint, linear forward map and sequential experiment loop
* `bound.py`: sPCE lower bound and closed-form Student-t marginal likelihood
* `bound_totalmi.py`: total mutual information variant `I(theta; h_T)`, used for the training-objective ablation
* `modules.py`: policy network with sufficient-statistics encoder, set encoder, emitter and static design baseline

**Training and evaluation**

* `training.py`: runs from the terminal with configurable arguments, saves the model and loss history to the output directory
* `evaluate_inter.py`: policy rollout and Gibbs posterior inference over main effects and interactions
* `run_eval.py`: evaluates a trained checkpoint and writes a structured `eval_summary.json`
* `run_eval_prior_mismatch.py`: same evaluation, with ground truth drawn from a different prior than training

**Baselines**

* `myopic_evaluate.py`: greedy one-step expected information gain rollout
* `run_eval_adaptive_baseline.py`: myopic baseline, wraps `myopic_evaluate.py`
* `run_eval_pb.py`: Plackett-Burman orthogonal fractional-factorial baseline
* `run_eval_distilled_sensing.py`: multi-stage adaptive baseline with thresholding between stages

## Launchers

SLURM array jobs. Adjust the `#SBATCH` header for your cluster.

* `launchers/run_case_studies.sh`: trains and evaluates the static and DABS arms for the three case studies
* `launchers/run_baselines_matched.sh`: evaluates the myopic, Plackett-Burman and multi-stage adaptive baselines
* `launchers/run_ablations.sh`: generates and submits the hyperparameter sensitivity, method ablation and robustness jobs

## Requirements

```
pip install -r requirements.txt
```

Python 3.9 or newer.

## Train

```
python training.py \
--num-factors 15 \
--num-experiments 16 \
--design-network-type dad \
--encoder-type suff \
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

```
python run_eval.py \
--model-path results/case1/model_20260101_120000.pt \
--num-rollouts 100 \
--num-gibbs-samples 200 \
--burnin 100 \
--threshold 0.5 \
--save-json results/case1/eval_summary.json
```

Reports F1, TPR and FDR for main effects and for two-factor interactions, together with
posterior inclusion probabilities and effect-size estimates.

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


