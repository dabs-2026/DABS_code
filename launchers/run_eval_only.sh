#!/bin/bash -l
#SBATCH --job-name=dabs_eval
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --array=0-379%32


module purge
module load PyTorch-bundle/2.6.0-foss-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs
cd $SLURM_SUBMIT_DIR
TASK=$SLURM_ARRAY_TASK_ID

ROOT_C2=${ROOT_C2:-/scratch/brussel/vo/000/bvo00004/vsc38578/dabs_5_2_hard_psweep}
ROOT_C3=${ROOT_C3:-/scratch/brussel/vo/000/bvo00004/vsc38578/dabs_5_3_focus}

P_C2=(10 12 14 15 16 18 20 22 24 26)
P_C3=(4 4 4 4 4 4 4 4 4 4  6 6 6 6 6 6 6 6 6  8 8 8 8 8 8 8 8 8)
T_C3=(3 4 5 6 8 10 12 16 18 20  6 8 10 12 14 16 18 20 24  6 8 10 12 14 16 18 20 24)

EVAL="--num-rollouts 100 --seed 42 --num-gibbs-samples 200 --burnin 100 --threshold 0.5"

if   [ $TASK -lt 50  ]; then
    CASE=2; ARM=static; i=$TASK
    SEED=$(( i%5+1 )); P=${P_C2[$(( i/5 ))]}; T=16
    DIR=$ROOT_C2/static_p${P}_seed${SEED}
elif [ $TASK -lt 100 ]; then
    CASE=2; ARM=dabs;   i=$(( TASK-50 ))
    SEED=$(( i%5+1 )); P=${P_C2[$(( i/5 ))]}; T=16
    DIR=$ROOT_C2/dabs_warm_suff_p${P}_seed${SEED}
elif [ $TASK -lt 240 ]; then
    CASE=3; ARM=static; i=$(( TASK-100 ))
    SEED=$(( i%5+1 )); c=$(( i/5 )); P=${P_C3[$c]}; T=${T_C3[$c]}
    DIR=$ROOT_C3/static_p${P}_T${T}_seed${SEED}
else
    CASE=3; ARM=dabs;   i=$(( TASK-240 ))
    SEED=$(( i%5+1 )); c=$(( i/5 )); P=${P_C3[$c]}; T=${T_C3[$c]}
    DIR=$ROOT_C3/dabs_warm_suff_p${P}_T${T}_seed${SEED}
fi

OUT="$DIR/eval_summary_mh.json"
if [ -f "$OUT" ]; then
    echo "[skip] $OUT already done"; exit 0
fi

CKPT=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "[FAIL] no checkpoint in $DIR"; exit 1
fi

echo "=== Case $CASE $ARM  p=$P T=$T seed=$SEED ==="
echo "    ckpt: $CKPT"

python run_eval.py --model-path "$CKPT" $EVAL --save-json "$OUT"
