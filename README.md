oed/primitives.py: Labels each random variable as latent parameter, experimental design or observation
oed/design.py: Connects the model, loss function and optimizer into a single training step
bound.py: sPCE lower bound, sNMC upper bound and REINFORCE gradient estimator with diversity penalty
modules.py: Permutation-invariant policy network with encoder, sum pooling, emitter and design baselines
screening_model.py: Spike-and-slab prior with heredity constraint, linear forward map and sequential experiment loop
training.py: Runs from the terminal with configurable arguments, saves model and loss history to results/
evaluate.py: Evaluation with Gibbs posterior inference and effect size estimates
evaluate_inter.py: Same as evaluate.py with additional reporting of interaction activity probabilities

Train: python training.py \
    --num-factors 15 \
    --num-experiments 16 \
    --num-steps 30000 \
    --lr 0.0003 \
    --gamma 0.999 \
    --encoding-dim 32 \
    --hidden-dim 128 \
    --num-outer-samples 200 \
    --num-inner-samples 100 \
    --adam-betas-wd 0.9 0.999 0.01

Evaluate: # Main effects only
python evaluate.py \
    --model-path results/checkpoint.pt \
    --num-gibbs-samples 2000 \
    --burnin 500 \
    --seed 1

# Main effects and interactions
python evaluate_inter.py \
    --model-path results/checkpoint.pt \
    --num-gibbs-samples 2000 \
    --burnin 500 \
    --seed 1
