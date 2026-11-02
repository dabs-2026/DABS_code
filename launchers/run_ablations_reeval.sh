#!/bin/bash -l
#SBATCH --job-name=dabs_abl_reeval
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ablreeval_%A_%a.out
#SBATCH --error=logs/ablreeval_%A_%a.err
#SBATCH --array=0-279%32


module purge
module load PyTorch-bundle/2.6.0-foss-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TASK=$SLURM_ARRAY_TASK_ID
ABL=${ABL:-/scratch/brussel/vo/000/bvo00004/vsc38578/dabs_ablations}

if [ $TASK -lt 200 ]; then
    mapfile -t DIRS < <(find "$ABL" -name "model_*.pt" -printf '%h\n' | sort -u)
    NDIRS=${#DIRS[@]}
    if [ $TASK -ge $NDIRS ]; then
        echo "[skip] task $TASK beyond $NDIRS checkpoint directories"; exit 0
    fi
    DIR=${DIRS[$TASK]}
    OUT="$DIR/eval_summary_mh.json"
    if [ -f "$OUT" ]; then echo "[skip] $OUT"; exit 0; fi

    CKPT=$(ls -t "$DIR"/model_*.pt | head -1)

    ROLL=$(python -c "import json;print(json.load(open('$DIR/eval_summary.json'))['eval_settings']['num_rollouts'])" 2>/dev/null || echo 50)
    SEED=$(python -c "import json;print(json.load(open('$DIR/eval_summary.json'))['eval_settings']['seed'])" 2>/dev/null || echo 42)

    OVERRIDE=""
    case "$(basename "$DIR")" in
        *heredity_weak_eval*) OVERRIDE="--eval-weak-heredity" ;;
    esac

    echo "=== ablation $(basename "$DIR")  rollouts=$ROLL seed=$SEED $OVERRIDE ==="
    python run_eval.py --model-path "$CKPT" \
        --num-rollouts "$ROLL" --seed "$SEED" \
        --num-gibbs-samples 200 --burnin 100 --threshold 0.5 \
        $OVERRIDE --save-json "$OUT"
    exit $?
fi

IDX=$(( TASK - 200 ))
BASELINE_BASE=$ABL/baseline_sensitivity
STATIC_DIR=$BASELINE_BASE/static_seed1
OUTBASE=$ABL/prior_mismatch

if [ $IDX -lt 50 ]; then
    STUDY=study1_sparsity; I=$IDX
    SEED=$(( (I % 5) + 1 ))
    ARM_IDX=$(( (I / 5) % 2 )); PI_IDX=$(( I / 10 ))
    PIS=(0.1 0.2 0.3 0.4 0.5); EVAL_PI=${PIS[$PI_IDX]}
    CELL_TAG=pi_${EVAL_PI}; OVERRIDE="--eval-expected-active $EVAL_PI"
else
    STUDY=study2_beta; I=$(( IDX - 50 ))
    SEED=$(( (I % 5) + 1 ))
    ARM_IDX=$(( (I / 5) % 2 )); BETA_IDX=$(( I / 10 ))
    BETA_AS=(2 4 6); BETA_BS=(8 6 4)
    A=${BETA_AS[$BETA_IDX]}; B=${BETA_BS[$BETA_IDX]}
    CELL_TAG=beta_${A}_${B}; OVERRIDE="--eval-pi-beta-a $A --eval-pi-beta-b $B"
fi

if [ $ARM_IDX -eq 0 ]; then
    ARM=dabs; CKPT_DIR=$BASELINE_BASE/dabs_baseline_seed${SEED}
else
    ARM=static; CKPT_DIR=$STATIC_DIR
fi

OUTDIR=$OUTBASE/${STUDY}/${ARM}_${CELL_TAG}_seed${SEED}
mkdir -p "$OUTDIR"
if [ -f "$OUTDIR/eval_summary_mh.json" ]; then echo "[skip] $OUTDIR"; exit 0; fi

CKPT=$(ls -t "$CKPT_DIR"/model_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "[FAIL] no ckpt at $CKPT_DIR"; exit 1; fi

echo "=== prior-mismatch $STUDY $ARM $CELL_TAG seed=$SEED ==="
python run_eval_prior_mismatch.py --model-path "$CKPT" \
    --num-rollouts 100 --seed 42 --num-gibbs-samples 200 \
    --burnin 100 --threshold 0.5 $OVERRIDE \
    --save-json "$OUTDIR/eval_summary_mh.json"
