#!/bin/bash -l
#SBATCH --job-name=dabs_cases
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=logs/case_%A_%a.out
#SBATCH --error=logs/case_%A_%a.err
#SBATCH --array=0-449%16

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null \
    || module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

mkdir -p logs
cd $SLURM_SUBMIT_DIR
TASK=$SLURM_ARRAY_TASK_ID

T_C1=(4 6 8 10 12 14 16)
P_C2=(10 12 14 15 16 18 20 22 24 26)
P_C3=(4 4 4 4 4 4 4 4 4 4  6 6 6 6 6 6 6 6 6  8 8 8 8 8 8 8 8 8)
T_C3=(3 4 5 6 8 10 12 16 18 20  6 8 10 12 14 16 18 20 24  6 8 10 12 14 16 18 20 24)

EVAL="--num-rollouts 100 --seed 42 --num-gibbs-samples 200 --burnin 100 --threshold 0.5"

if   [ $TASK -lt 35  ]; then CASE=1; ARM=static; i=$TASK;         SEED=$(( i%5+1 )); P=15; T=${T_C1[$(( i/5 ))]}
elif [ $TASK -lt 70  ]; then CASE=1; ARM=dabs;   i=$(( TASK-35 )); SEED=$(( i%5+1 )); P=15; T=${T_C1[$(( i/5 ))]}
elif [ $TASK -lt 120 ]; then CASE=2; ARM=static; i=$(( TASK-70 )); SEED=$(( i%5+1 )); P=${P_C2[$(( i/5 ))]}; T=16
elif [ $TASK -lt 170 ]; then CASE=2; ARM=dabs;   i=$(( TASK-120 ));SEED=$(( i%5+1 )); P=${P_C2[$(( i/5 ))]}; T=16
elif [ $TASK -lt 310 ]; then CASE=3; ARM=static; i=$(( TASK-170 ));SEED=$(( i%5+1 )); c=$(( i/5 )); P=${P_C3[$c]}; T=${T_C3[$c]}
else                         CASE=3; ARM=dabs;   i=$(( TASK-310 ));SEED=$(( i%5+1 )); c=$(( i/5 )); P=${P_C3[$c]}; T=${T_C3[$c]}
fi

case $CASE in
 1) MODEL="--noise-scale 0.5 --scale-active 2.0 --expected-active 0.2 --sigma2-beta0 1.0 --pi-beta-a 2 --pi-beta-b 8"
    HEREDITY="--strong-heredity"
    OUTBASE=dabs_curve_p15
    STATIC_DIR=$OUTBASE/static_T${T}_seed${SEED}
    DABS_DIR=$OUTBASE/dabs_original_T${T}_seed${SEED}
    DABS_ENC=original ;;
 2) MODEL="--noise-scale 0.5 --scale-active 1.0 --expected-active 0.2 --sigma2-beta0 1.0 --pi-beta-a 4 --pi-beta-b 6"
    HEREDITY="--strong-heredity"
    OUTBASE=dabs_5_2_hard_psweep
    STATIC_DIR=$OUTBASE/static_p${P}_seed${SEED}
    DABS_DIR=$OUTBASE/dabs_warm_suff_p${P}_seed${SEED}
    DABS_ENC=suff ;;
 3) MODEL="--noise-scale 0.25 --scale-active 2.0 --expected-active 0.5 --sigma2-beta0 0.25 --pi-beta-a 5 --pi-beta-b 5"
    HEREDITY="--strong-heredity"
    OUTBASE=dabs_5_3_focus
    STATIC_DIR=$OUTBASE/static_p${P}_T${T}_seed${SEED}
    DABS_DIR=$OUTBASE/dabs_warm_suff_p${P}_T${T}_seed${SEED}
    DABS_ENC=suff ;;
esac

COMMON="--num-factors $P --num-experiments $T $MODEL $HEREDITY \
        --scale-inactive 0.01 --sigma2-alpha0 3.0 --use-g-prior --pi-interaction 0.5 \
        --mlp-depth 1 --hidden-dim 64 --adam-betas-wd 0.9 0.999 0.0 \
        --grad-clip-max-norm 1.0 --log-every 100 --seed $SEED --device cuda"

if [ "$ARM" = "static" ]; then
    mkdir -p "$STATIC_DIR"
    if [ -f "$STATIC_DIR/eval_summary.json" ]; then echo "[skip] $STATIC_DIR"; exit 0; fi
    if [ $CASE -eq 2 ]; then OUTER=500;  INNER=1500
    else                     OUTER=2000; INNER=6000; fi
    echo "=== Case $CASE STATIC p=$P T=$T seed=$SEED ==="
    python training.py $COMMON \
        --design-network-type static --encoder-type original \
        --encoding-dim 16 --proj-hidden 64 --pe-dim 4 \
        --num-steps 2000 --lr 5e-3 --gamma 0.98 \
        --num-outer-samples $OUTER --num-inner-samples $INNER \
        --logit-l2-penalty 0.0 --diversity-penalty 0.0 \
        --tau-start 1.0 --tau-end 0.1 --tau-anneal-frac 0.5 \
        --output-dir "$STATIC_DIR"
    CKPT=$(ls -t "$STATIC_DIR"/model_*.pt 2>/dev/null | head -1)
    [ -n "$CKPT" ] && python run_eval.py --model-path "$CKPT" $EVAL \
        --save-json "$STATIC_DIR/eval_summary.json"
    exit 0
fi

mkdir -p "$DABS_DIR"
if [ -f "$DABS_DIR/eval_summary.json" ]; then echo "[skip] $DABS_DIR"; exit 0; fi
if [ $CASE -eq 2 ]; then INNER=256; else INNER=512; fi

WARM=""
if [ $CASE -ne 1 ]; then
    WCKPT=$(ls -t "$STATIC_DIR"/model_*.pt 2>/dev/null | head -1)
    if [ -z "$WCKPT" ]; then
        echo "[FAIL] missing static checkpoint for warm start: $STATIC_DIR"; exit 1
    fi
    WARM="--warm-start-ckpt $WCKPT"
fi

echo "=== Case $CASE DABS p=$P T=$T seed=$SEED enc=$DABS_ENC ==="
python training.py $COMMON \
    --design-network-type dad --encoder-type $DABS_ENC \
    --encoding-dim 32 --proj-hidden 32 --pe-dim 0 \
    --num-steps 30000 --lr 1e-3 --gamma 0.95 \
    --num-outer-samples 6000 --num-inner-samples $INNER \
    --logit-l2-penalty 0.01 --diversity-penalty 0.0 \
    --tau-start 1.0 --tau-end 0.5 --tau-anneal-frac 0.8 \
    $WARM --output-dir "$DABS_DIR"

CKPT=$(ls -t "$DABS_DIR"/model_*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] && python run_eval.py --model-path "$CKPT" $EVAL \
    --save-json "$DABS_DIR/eval_summary.json"
