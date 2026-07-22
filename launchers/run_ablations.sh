#!/bin/bash

set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Writing submit_abl_baseline_sensitivity_vub.sh ==="
cat > submit_abl_baseline_sensitivity_vub.sh << 'SBATCH_SENS'
#!/bin/bash -l
#SBATCH --job-name=abl_bs
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/abl_bs_%A_%a.out
#SBATCH --error=logs/abl_bs_%A_%a.err
#SBATCH --array=0-137%8

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null \
    || module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

mkdir -p logs
TASK=$SLURM_ARRAY_TASK_ID
OUTBASE=$VSC_SCRATCH/dabs_ablations/baseline_sensitivity
mkdir -p "$OUTBASE"

P=10; T=8; NOISE=0.5; SCALE_A=1.0; SCALE_I=0.01; PI_A=2; PI_B=8
LR=1e-3; NSTEPS=30000; GAMMA=0.95; HIDDEN=64; ENC=32
LOGIT_L2=0.01; DIV=0.0; TAU_S=1.0; TAU_E=0.3; ANNEAL=0.5
WD=0.0; NUM_OUTER=251; NUM_INNER=512
STATIC_DIR=$OUTBASE/static_seed1

if [ $TASK -eq 0 ]; then
    mkdir -p "$STATIC_DIR"
    if [ -f "$STATIC_DIR/eval_summary.json" ]; then echo "[skip]"; exit 0; fi
    cd $SLURM_SUBMIT_DIR
    python training.py --num-factors $P --num-experiments $T \
        --noise-scale $NOISE --expected-active 0.2 \
        --scale-active $SCALE_A --scale-inactive $SCALE_I \
        --sigma2-alpha0 3.0 --use-g-prior --strong-heredity \
        --pi-interaction 0.5 --pi-beta-a $PI_A --pi-beta-b $PI_B \
        --mlp-depth 1 --lr $LR --hidden-dim $HIDDEN \
        --encoding-dim $ENC --proj-hidden 32 --pe-dim 0 \
        --num-outer-samples $NUM_OUTER --num-inner-samples $NUM_INNER \
        --adam-betas-wd 0.9 0.999 $WD --grad-clip-max-norm 1.0 --log-every 50 \
        --seed 1 --device cuda --output-dir $STATIC_DIR \
        --design-network-type static --num-steps 15000 --gamma 1.0 \
        --early-stop --early-stop-patience 30 \
        --early-stop-min-delta 0.005 --early-stop-min-steps-frac 0.5
    CKPT=$(ls -t "$STATIC_DIR"/model_*.pt 2>/dev/null | head -1)
    python run_eval.py --model-path "$CKPT" --num-rollouts 50 --seed 42 \
        --num-gibbs-samples 200 --burnin 100 --threshold 0.5 \
        --save-json "$STATIC_DIR/eval_summary.json"
    exit 0
fi

WARM_CKPT=$(ls -t "$STATIC_DIR"/model_*.pt 2>/dev/null | head -1)
if [ -z "$WARM_CKPT" ]; then echo "[FAIL] no warm-start"; exit 1; fi

if [ $TASK -ge 1 ] && [ $TASK -le 12 ]; then
    SEED=$TASK; CELL="baseline"
else
    IDX=$(( TASK - 13 )); SEED=$(( (IDX % 5) + 1 )); CELL_IDX=$(( IDX / 5 ))
    CELLS=("lr_1e-4" "lr_5e-4" "lr_2e-3" "lr_5e-3" "steps_15k" "steps_60k"
           "decay_0.90" "decay_0.98" "wd_1e-4" "wd_1e-3" "hidden_32" "hidden_128"
           "enc_16" "enc_64" "l2_0.001" "l2_0.05" "l2_0.1" "div_0.01" "div_0.1"
           "taus_0.5" "taus_2.0" "taue_0.1" "taue_0.5" "anneal_0.3" "anneal_0.8")
    CELL=${CELLS[$CELL_IDX]}
    case $CELL in
        lr_1e-4)    LR=1e-4 ;;    lr_5e-4)    LR=5e-4 ;;
        lr_2e-3)    LR=2e-3 ;;    lr_5e-3)    LR=5e-3 ;;
        steps_15k)  NSTEPS=15000 ;;  steps_60k)  NSTEPS=60000 ;;
        decay_0.90) GAMMA=0.90 ;;   decay_0.98) GAMMA=0.98 ;;
        wd_1e-4)    WD=1e-4 ;;    wd_1e-3)    WD=1e-3 ;;
        hidden_32)  HIDDEN=32 ;;   hidden_128) HIDDEN=128 ;;
        enc_16)     ENC=16 ;;    enc_64)     ENC=64 ;;
        l2_0.001)   LOGIT_L2=0.001 ;; l2_0.05) LOGIT_L2=0.05 ;; l2_0.1) LOGIT_L2=0.1 ;;
        div_0.01)   DIV=0.01 ;;   div_0.1)    DIV=0.1 ;;
        taus_0.5)   TAU_S=0.5 ;;  taus_2.0)   TAU_S=2.0 ;;
        taue_0.1)   TAU_E=0.1 ;;  taue_0.5)   TAU_E=0.5 ;;
        anneal_0.3) ANNEAL=0.3 ;; anneal_0.8) ANNEAL=0.8 ;;
    esac
fi

OUTDIR=$OUTBASE/dabs_${CELL}_seed${SEED}
mkdir -p "$OUTDIR"
if [ -f "$OUTDIR/eval_summary.json" ]; then echo "[skip]"; exit 0; fi
cd $SLURM_SUBMIT_DIR

python training.py --num-factors $P --num-experiments $T \
    --noise-scale $NOISE --expected-active 0.2 \
    --scale-active $SCALE_A --scale-inactive $SCALE_I \
    --sigma2-alpha0 3.0 --use-g-prior --strong-heredity \
    --pi-interaction 0.5 --pi-beta-a $PI_A --pi-beta-b $PI_B \
    --mlp-depth 1 --lr $LR --hidden-dim $HIDDEN \
    --encoding-dim $ENC --proj-hidden 32 --pe-dim 0 \
    --num-outer-samples $NUM_OUTER --num-inner-samples $NUM_INNER \
    --adam-betas-wd 0.9 0.999 $WD --grad-clip-max-norm 1.0 --log-every 50 \
    --seed $SEED --device cuda --output-dir $OUTDIR \
    --design-network-type dad --encoder-type suff \
    --warm-start-ckpt "$WARM_CKPT" \
    --num-steps $NSTEPS --gamma $GAMMA \
    --tau-start $TAU_S --tau-end $TAU_E --tau-anneal-frac $ANNEAL \
    --diversity-penalty $DIV --logit-l2-penalty $LOGIT_L2 \
    --early-stop --early-stop-patience 30 \
    --early-stop-min-delta 0.005 --early-stop-min-steps-frac 0.5

CKPT=$(ls -t "$OUTDIR"/model_*.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    python run_eval.py --model-path "$CKPT" --num-rollouts 50 --seed 42 \
        --num-gibbs-samples 200 --burnin 100 --threshold 0.5 \
        --save-json "$OUTDIR/eval_summary.json"
fi
SBATCH_SENS

echo "=== Writing submit_abl_method_ablations_vub.sh ==="
cat > submit_abl_method_ablations_vub.sh << 'SBATCH_METH'
#!/bin/bash -l
#SBATCH --job-name=abl_meth
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/abl_meth_%A_%a.out
#SBATCH --error=logs/abl_meth_%A_%a.err
#SBATCH --array=0-54%8

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null \
    || module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

mkdir -p logs
TASK=$SLURM_ARRAY_TASK_ID
OUTBASE_BASE=$VSC_SCRATCH/dabs_ablations/baseline_sensitivity
STATIC_DIR=$OUTBASE_BASE/static_seed1
OUTBASE=$VSC_SCRATCH/dabs_ablations/method_ablations
mkdir -p "$OUTBASE"

SEED=$(( (TASK % 5) + 1 )); CELL_IDX=$(( TASK / 5 ))
CELLS=(encoder_original init_random reward_dense reg_none reg_cosine reg_both mi_total sigma_0.25 sigma_1.0 sigma_2.0 heredity_weak_eval)
CELL=${CELLS[$CELL_IDX]}

OUTDIR=$OUTBASE/dabs_${CELL}_seed${SEED}
mkdir -p "$OUTDIR"
if [ -f "$OUTDIR/eval_summary.json" ]; then echo "[skip]"; exit 0; fi

P=10; T=8; NOISE=0.5; LR=1e-3; NSTEPS=30000; GAMMA=0.95
HIDDEN=64; ENC=32; LOGIT_L2=0.01; DIV=0.0
TAU_S=1.0; TAU_E=0.3; ANNEAL=0.5
ENCODER=suff; REWARD_FLAG=""; INIT_FLAG=""
REGULARISATION_EXTRA=""; MI_TARGET_FLAG=""; EVAL_HEREDITY_OVERRIDE=""
WARM_CKPT=$(ls -t "$STATIC_DIR"/model_*.pt 2>/dev/null | head -1)

case $CELL in
    encoder_original) ENCODER=original ;;
    init_random) WARM_CKPT="" ;;
    reward_dense) REWARD_FLAG="--dense-rewards" ;;
    reg_none) LOGIT_L2=0.0 ;;
    reg_cosine) LOGIT_L2=0.0; REGULARISATION_EXTRA="--cosine-penalty 0.01" ;;
    reg_both) REGULARISATION_EXTRA="--cosine-penalty 0.01" ;;
    mi_total) MI_TARGET_FLAG="--use-total-mi" ;;
    sigma_0.25) NOISE=0.25 ;;
    sigma_1.0) NOISE=1.0 ;;
    sigma_2.0) NOISE=2.0 ;;
    heredity_weak_eval) EVAL_HEREDITY_OVERRIDE="--eval-weak-heredity" ;;
esac

if [ -n "$WARM_CKPT" ]; then INIT_FLAG="--warm-start-ckpt $WARM_CKPT"; fi
cd $SLURM_SUBMIT_DIR

python training.py --num-factors $P --num-experiments $T \
    --noise-scale $NOISE --expected-active 0.2 \
    --scale-active 1.0 --scale-inactive 0.01 \
    --sigma2-alpha0 3.0 --use-g-prior --strong-heredity \
    --pi-interaction 0.5 --pi-beta-a 2 --pi-beta-b 8 \
    --mlp-depth 1 --lr $LR --hidden-dim $HIDDEN \
    --encoding-dim $ENC --proj-hidden 32 --pe-dim 0 \
    --num-outer-samples 251 --num-inner-samples 512 \
    --adam-betas-wd 0.9 0.999 0.0 --grad-clip-max-norm 1.0 --log-every 50 \
    --seed $SEED --device cuda --output-dir $OUTDIR \
    --design-network-type dad --encoder-type $ENCODER $INIT_FLAG \
    --num-steps $NSTEPS --gamma $GAMMA \
    --tau-start $TAU_S --tau-end $TAU_E --tau-anneal-frac $ANNEAL \
    --diversity-penalty $DIV --logit-l2-penalty $LOGIT_L2 \
    $REWARD_FLAG $MI_TARGET_FLAG $REGULARISATION_EXTRA \
    --early-stop --early-stop-patience 30 \
    --early-stop-min-delta 0.005 --early-stop-min-steps-frac 0.5

CKPT=$(ls -t "$OUTDIR"/model_*.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    python run_eval.py --model-path "$CKPT" --num-rollouts 50 --seed 42 \
        --num-gibbs-samples 200 --burnin 100 --threshold 0.5 \
        $EVAL_HEREDITY_OVERRIDE --save-json "$OUTDIR/eval_summary.json"
fi
SBATCH_METH

echo "=== Writing submit_abl_robustness_vub.sh ==="
cat > submit_abl_robustness_vub.sh << 'SBATCH_ROB'
#!/bin/bash -l
#SBATCH --job-name=abl_rob
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=logs/abl_rob_%A_%a.out
#SBATCH --error=logs/abl_rob_%A_%a.err
#SBATCH --array=0-79%8

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null \
    || module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

mkdir -p logs
TASK=$SLURM_ARRAY_TASK_ID
BASELINE_BASE=$VSC_SCRATCH/dabs_ablations/baseline_sensitivity
STATIC_DIR=$BASELINE_BASE/static_seed1
OUTBASE=$VSC_SCRATCH/dabs_ablations/prior_mismatch
mkdir -p "$OUTBASE"

if [ $TASK -lt 50 ]; then
    STUDY=study1_sparsity; IDX=$TASK
    SEED=$(( (IDX % 5) + 1 ))
    ARM_IDX=$(( (IDX / 5) % 2 )); PI_IDX=$(( IDX / 10 ))
    PIS=(0.1 0.2 0.3 0.4 0.5); EVAL_PI=${PIS[$PI_IDX]}
    if [ $ARM_IDX -eq 0 ]; then
        ARM=dabs; CKPT_DIR=$BASELINE_BASE/dabs_baseline_seed${SEED}
    else
        ARM=static; CKPT_DIR=$STATIC_DIR
    fi
    CELL_TAG=pi_${EVAL_PI}; OVERRIDE="--eval-expected-active $EVAL_PI"
else
    STUDY=study2_beta; IDX=$(( TASK - 50 ))
    SEED=$(( (IDX % 5) + 1 ))
    ARM_IDX=$(( (IDX / 5) % 2 )); BETA_IDX=$(( IDX / 10 ))
    BETA_AS=(2 4 6); BETA_BS=(8 6 4)
    A=${BETA_AS[$BETA_IDX]}; B=${BETA_BS[$BETA_IDX]}
    if [ $ARM_IDX -eq 0 ]; then
        ARM=dabs; CKPT_DIR=$BASELINE_BASE/dabs_baseline_seed${SEED}
    else
        ARM=static; CKPT_DIR=$STATIC_DIR
    fi
    CELL_TAG=beta_${A}_${B}; OVERRIDE="--eval-pi-beta-a $A --eval-pi-beta-b $B"
fi

OUTDIR=$OUTBASE/${STUDY}/${ARM}_${CELL_TAG}_seed${SEED}
mkdir -p "$OUTDIR"
if [ -f "$OUTDIR/eval_summary.json" ]; then echo "[skip]"; exit 0; fi

CKPT=$(ls -t "$CKPT_DIR"/model_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "[FAIL] no ckpt at $CKPT_DIR"; exit 1; fi
cd $SLURM_SUBMIT_DIR

python run_eval_prior_mismatch.py --model-path "$CKPT" \
    --num-rollouts 100 --seed 42 --num-gibbs-samples 200 \
    --burnin 100 --threshold 0.5 $OVERRIDE \
    --save-json "$OUTDIR/eval_summary.json"
SBATCH_ROB

echo "=== Writing submit_5_2_longtrain_vub.sh ==="
cat > submit_5_2_longtrain_vub.sh << 'SBATCH_LT'
#!/bin/bash -l
#SBATCH --job-name=52_lt
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=18:00:00
#SBATCH --output=logs/52lt_%A_%a.out
#SBATCH --error=logs/52lt_%A_%a.err
#SBATCH --array=0-23%8

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null \
    || module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

mkdir -p logs
TASK=$SLURM_ARRAY_TASK_ID
OUTBASE=$VSC_SCRATCH/dabs_5_2_longtrain
mkdir -p "$OUTBASE"
P_VALUES=(14 18 22 26); T=16

if [ $TASK -lt 4 ]; then
    P=${P_VALUES[$TASK]}
    STATIC_DIR=$OUTBASE/static_p${P}_seed1
    mkdir -p "$STATIC_DIR"
    if [ -f "$STATIC_DIR/eval_summary.json" ]; then echo "[skip]"; exit 0; fi
    cd $SLURM_SUBMIT_DIR
    python training.py --num-factors $P --num-experiments $T \
        --noise-scale 0.5 --expected-active 0.5 \
        --scale-active 1.0 --scale-inactive 0.01 \
        --sigma2-alpha0 3.0 --use-g-prior --strong-heredity \
        --pi-interaction 0.5 --pi-beta-a 4 --pi-beta-b 6 \
        --mlp-depth 1 --lr 1e-3 --hidden-dim 64 \
        --encoding-dim 32 --proj-hidden 32 --pe-dim 0 \
        --num-outer-samples 251 --num-inner-samples 512 \
        --adam-betas-wd 0.9 0.999 0.0 --grad-clip-max-norm 1.0 --log-every 50 \
        --seed 1 --device cuda --output-dir $STATIC_DIR \
        --design-network-type static --num-steps 15000 --gamma 1.0 \
        --early-stop --early-stop-patience 30 \
        --early-stop-min-delta 0.005 --early-stop-min-steps-frac 0.5
    CKPT=$(ls -t "$STATIC_DIR"/model_*.pt 2>/dev/null | head -1)
    python run_eval.py --model-path "$CKPT" --num-rollouts 100 --seed 42 \
        --num-gibbs-samples 200 --burnin 100 --threshold 0.5 \
        --save-json "$STATIC_DIR/eval_summary.json"
    exit 0
fi

IDX=$(( TASK - 4 )); SEED=$(( (IDX % 5) + 1 )); P_IDX=$(( IDX / 5 ))
P=${P_VALUES[$P_IDX]}
STATIC_DIR=$OUTBASE/static_p${P}_seed1
OUTDIR=$OUTBASE/dabs_90k_p${P}_seed${SEED}
mkdir -p "$OUTDIR"
if [ -f "$OUTDIR/eval_summary.json" ]; then echo "[skip]"; exit 0; fi
WARM_CKPT=$(ls -t "$STATIC_DIR"/model_*.pt 2>/dev/null | head -1)
if [ -z "$WARM_CKPT" ]; then echo "[FAIL] no static ckpt"; exit 1; fi
cd $SLURM_SUBMIT_DIR

python training.py --num-factors $P --num-experiments $T \
    --noise-scale 0.5 --expected-active 0.5 \
    --scale-active 1.0 --scale-inactive 0.01 \
    --sigma2-alpha0 3.0 --use-g-prior --strong-heredity \
    --pi-interaction 0.5 --pi-beta-a 4 --pi-beta-b 6 \
    --mlp-depth 1 --lr 1e-3 --hidden-dim 64 \
    --encoding-dim 32 --proj-hidden 32 --pe-dim 0 \
    --num-outer-samples 251 --num-inner-samples 512 \
    --adam-betas-wd 0.9 0.999 0.0 --grad-clip-max-norm 1.0 --log-every 50 \
    --seed $SEED --device cuda --output-dir $OUTDIR \
    --design-network-type dad --encoder-type suff \
    --warm-start-ckpt "$WARM_CKPT" \
    --num-steps 90000 --gamma 0.95 \
    --tau-start 1.0 --tau-end 0.3 --tau-anneal-frac 0.5 \
    --diversity-penalty 0.0 --logit-l2-penalty 0.01 \
    --early-stop --early-stop-patience 30 \
    --early-stop-min-delta 0.005 --early-stop-min-steps-frac 0.5

CKPT=$(ls -t "$OUTDIR"/model_*.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    python run_eval.py --model-path "$CKPT" --num-rollouts 100 --seed 42 \
        --num-gibbs-samples 200 --burnin 100 --threshold 0.5 \
        --save-json "$OUTDIR/eval_summary.json"
fi
SBATCH_LT

echo ""
echo "=== All 4 sbatch scripts written to disk ==="
ls -la submit_*_vub.sh | tail -5
echo ""
echo "=== Submitting jobs ==="
J_SENS=$(sbatch --parsable submit_abl_baseline_sensitivity_vub.sh)
echo "  sensitivity: $J_SENS"
J_LT=$(sbatch --parsable submit_5_2_longtrain_vub.sh)
echo "  longtrain:   $J_LT"
J_METH=$(sbatch --parsable --dependency=afterany:$J_SENS submit_abl_method_ablations_vub.sh)
echo "  method:      $J_METH  (waits for $J_SENS)"
J_ROB=$(sbatch --parsable --dependency=afterany:$J_SENS submit_abl_robustness_vub.sh)
echo "  robustness:  $J_ROB   (waits for $J_SENS)"

echo ""
echo "=== Queue ==="
squeue -u $USER
