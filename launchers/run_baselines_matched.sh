#!/bin/bash -l
#SBATCH --job-name=dabs_base
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/base_%A_%a.out
#SBATCH --error=logs/base_%A_%a.err
#SBATCH --array=0-334%16

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null \
    || module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TASK=$SLURM_ARRAY_TASK_ID
OUTROOT=${OUTROOT:-dabs_baselines}

T_C1=(4 6 8 10 12 14 16)
P_C2=(10 12 14 15 16 18 20 22 24 26)
P_C2_PB=(10 12 14 15)
P_C3=(4 4 4 4 4 4 4 4 4 4  6 6 6 6 6 6 6 6 6  8 8 8 8 8 8 8 8 8)
T_C3=(3 4 5 6 8 10 12 16 18 20  6 8 10 12 14 16 18 20 24  6 8 10 12 14 16 18 20 24)

ROLLOUTS=100
GIBBS="--num-gibbs-samples 200 --burnin 100 --threshold 0.5"
COMMON="--scale-inactive 0.01 --sigma2-alpha0 3.0 --pi-interaction 0.5 --use-g-prior"

if   [ $TASK -lt 35 ]; then
    SECTION=5_1; ARM=myopic
    IDX=$TASK;              SEED=$(( (IDX % 5) + 1 )); P=15; T=${T_C1[$(( IDX / 5 ))]}
elif [ $TASK -lt 40 ]; then
    SECTION=5_1; ARM=pb
    IDX=$(( TASK - 35 ));   SEED=$(( (IDX % 5) + 1 )); P=15; T=16
elif [ $TASK -lt 75 ]; then
    SECTION=5_1; ARM=ds
    IDX=$(( TASK - 40 ));   SEED=$(( (IDX % 5) + 1 )); P=15; T=${T_C1[$(( IDX / 5 ))]}
elif [ $TASK -lt 125 ]; then
    SECTION=5_2; ARM=myopic
    IDX=$(( TASK - 75 ));   SEED=$(( (IDX % 5) + 1 )); P=${P_C2[$(( IDX / 5 ))]}; T=16
elif [ $TASK -lt 145 ]; then
    SECTION=5_2; ARM=pb
    IDX=$(( TASK - 125 ));  SEED=$(( (IDX % 5) + 1 )); P=${P_C2_PB[$(( IDX / 5 ))]}; T=16
elif [ $TASK -lt 195 ]; then
    SECTION=5_2; ARM=ds
    IDX=$(( TASK - 145 ));  SEED=$(( (IDX % 5) + 1 )); P=${P_C2[$(( IDX / 5 ))]}; T=16
else
    SECTION=5_3; ARM=myopic
    IDX=$(( TASK - 195 ));  SEED=$(( (IDX % 5) + 1 ))
    CELL=$(( IDX / 5 )); P=${P_C3[$CELL]}; T=${T_C3[$CELL]}
fi

case $SECTION in
  5_1)
        MODEL="--noise-scale 0.5 --scale-active 2.0 --expected-active 0.2 --sigma2-beta0 1.0 --pi-beta-a 2 --pi-beta-b 8"
        HEREDITY="--strong-heredity" ;;
  5_2)
        MODEL="--noise-scale 0.5 --scale-active 1.0 --expected-active 0.2 --sigma2-beta0 1.0 --pi-beta-a 4 --pi-beta-b 6"
        HEREDITY="--strong-heredity" ;;
  5_3)
        MODEL="--noise-scale 0.25 --scale-active 2.0 --expected-active 0.5 --sigma2-beta0 0.25 --pi-beta-a 5 --pi-beta-b 5"
        HEREDITY="--strong-heredity" ;;
esac

OUTDIR=$OUTROOT/$SECTION/${ARM}_p${P}_T${T}_seed${SEED}
mkdir -p "$OUTDIR"
if [ -f "$OUTDIR/eval_summary.json" ]; then
    echo "[skip] $OUTDIR already done"; exit 0
fi

echo "=== $SECTION $ARM  p=$P T=$T seed=$SEED ==="
echo "    model: $MODEL $HEREDITY"

case $ARM in
  myopic)
    python run_eval_adaptive_baseline.py \
        --num-factors $P --num-experiments $T \
        $MODEL $HEREDITY $COMMON \
        --num-candidates 1000 \
        --num-rollouts $ROLLOUTS --seed $SEED $GIBBS \
        --save-json "$OUTDIR/eval_summary.json"
    ;;
  pb)
    python run_eval_pb.py \
        --num-factors $P --num-experiments $T \
        $MODEL $HEREDITY $COMMON \
        --num-rollouts $ROLLOUTS --seed $SEED $GIBBS \
        --save-json "$OUTDIR/eval_summary.json"
    ;;
  ds)
    python run_eval_distilled_sensing.py \
        --num-factors $P --num-experiments $T \
        $MODEL $HEREDITY $COMMON \
        --num-stages 3 --keep-fraction 0.5 \
        --num-rollouts $ROLLOUTS --seed $SEED \
        --save-json "$OUTDIR/eval_summary.json"
    ;;
esac
