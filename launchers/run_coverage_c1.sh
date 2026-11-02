#!/bin/bash -l
#SBATCH --job-name=dabs_cov1
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cov1_%A_%a.out
#SBATCH --error=logs/cov1_%A_%a.err
#SBATCH --array=0-139%32


module purge
module load PyTorch-bundle/2.6.0-foss-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TASK=$SLURM_ARRAY_TASK_ID

CKPT_ROOT=""
for R in /scratch/brussel/vo/000/bvo00004/vsc38578/dabs_curve_p15 \
         $HOME/dabs_curve_p15 \
         ./dabs_curve_p15; do
    if [ -d "$R" ]; then CKPT_ROOT=$R; break; fi
done
if [ -z "$CKPT_ROOT" ]; then
    echo "[FAIL] dabs_curve_p15 not found on scratch, home or submit dir"; exit 1
fi

LEVELS=(0.50 0.80 0.90 0.95)
T_LIST=(4 6 8 10 12 14 16)

LEVEL=${LEVELS[$(( TASK / 35 ))]}
SUB=$(( TASK % 35 ))
SEED=$(( SUB % 5 + 1 ))
T=${T_LIST[$(( SUB / 5 ))]}

OUTROOT=dabs_coverage_c1_ord_${LEVEL}
CKPT_DIR=$CKPT_ROOT/dabs_original_T${T}_seed${SEED}
OUTDIR=$OUTROOT/T${T}_seed${SEED}
mkdir -p "$OUTDIR"

if [ -f "$OUTDIR/coverage.json" ]; then
    echo "[skip] $OUTDIR already done"; exit 0
fi

CKPT=$(ls -t "$CKPT_DIR"/model_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "[FAIL] no checkpoint found in $CKPT_DIR"; exit 1
fi

echo "=== coverage  p=15 T=$T seed=$SEED  level=$LEVEL ==="
echo "    ckpt: $CKPT"

python run_coverage.py \
    --model-path "$CKPT" \
    --num-rollouts 100 \
    --seed 42 \
    --num-gibbs-samples 200 \
    --burnin 100 \
    --threshold 0.5 \
    --level $LEVEL \
    --save-json "$OUTDIR/coverage.json"
