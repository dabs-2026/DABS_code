#!/bin/bash -l
#SBATCH --job-name=dabs_cov3
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cov3_%A_%a.out
#SBATCH --error=logs/cov3_%A_%a.err
#SBATCH --array=0-179%32


module purge
module load PyTorch-bundle/2.6.0-foss-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TASK=$SLURM_ARRAY_TASK_ID
CKPT_ROOT=/scratch/brussel/vo/000/bvo00004/vsc38578/dabs_5_3_focus
P=8

LEVELS=(0.50 0.80 0.90 0.95)
T_LIST=(6 8 10 12 14 16 18 20 24)

LEVEL=${LEVELS[$(( TASK / 45 ))]}
SUB=$(( TASK % 45 ))
SEED=$(( SUB % 5 + 1 ))
T=${T_LIST[$(( SUB / 5 ))]}

OUTROOT=dabs_coverage_c3_ord_${LEVEL}
CKPT_DIR=$CKPT_ROOT/dabs_warm_suff_p${P}_T${T}_seed${SEED}
OUTDIR=$OUTROOT/p${P}_T${T}_seed${SEED}
mkdir -p "$OUTDIR"

if [ -f "$OUTDIR/coverage.json" ]; then
    echo "[skip] $OUTDIR already done"; exit 0
fi

CKPT=$(ls -t "$CKPT_DIR"/model_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "[FAIL] no checkpoint found in $CKPT_DIR"; exit 1
fi

echo "=== coverage  p=$P T=$T seed=$SEED  level=$LEVEL ==="
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
