module load cluster/wice/gpu_h100
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

mkdir -p results_master \
         results_5_1_gprior results_5_1_static_gprior \
         results_5_2_gprior results_5_2_static_gprior \
         results_5_3_dethered_gprior results_5_3_strongh_gprior \
         results_baselines
rm -rf __pycache__ 2>/dev/null || true

echo "Start: $(date)"

count_ckpts() {
  local outdir=$1 p=$2 T=$3 type=$4
  python3 - <<PYEOF
import torch, glob
n=0
for pt in glob.glob('$outdir/model_*.pt'):
    try:
        c=torch.load(pt, map_location='cpu', weights_only=False); pp=c['params']
        if pp['num_factors']==$p and pp['T']==$T and pp['design_network_type']=='$type':
            n+=1
    except: pass
print(n)
PYEOF
}

train_dabs() {
  local outdir=$1 p=$2 T=$3 s=$4; shift 4
  local hidden=64 outer=4000 steps=3000
  if [ "$p" -eq 20 ]; then hidden=128; outer=2000; steps=5000; fi
  python3 training.py \
    --num-factors $p --num-experiments $T \
    --noise-scale 0.5 --expected-active 0.2 \
    --scale-active 2.0 --scale-inactive 0.01 --use-g-prior \
    --lr 1e-3 --num-steps $steps --gamma 0.98 --adam-betas-wd 0.9 0.999 0.0 \
    --hidden-dim $hidden --encoding-dim 32 \
    --num-outer-samples $outer --num-inner-samples 512 \
    --tau-start 1.0 --tau-end 0.5 --tau-anneal-frac 0.5 \
    --diversity-penalty 0.0 --logit-l2-penalty 0.01 \
    --encoder-type suff --design-network-type dad \
    --seed $s --device cuda --output-dir "$outdir" "$@" || true
}

train_static() {
  local outdir=$1 p=$2 T=$3 s=$4; shift 4
  python3 training.py \
    --num-factors $p --num-experiments $T \
    --noise-scale 0.5 --expected-active 0.2 \
    --scale-active 2.0 --scale-inactive 0.01 --use-g-prior \
    --lr 5e-3 --num-steps 2000 --gamma 0.98 --adam-betas-wd 0.9 0.999 0.0 \
    --hidden-dim 64 --encoding-dim 32 \
    --num-outer-samples 2000 --num-inner-samples 512 \
    --tau-start 1.0 --tau-end 0.5 --tau-anneal-frac 0.5 \
    --diversity-penalty 0.0 --logit-l2-penalty 0.0 \
    --design-network-type static \
    --seed $s --device cuda --output-dir "$outdir" "$@" || true
}

train_if_needed() {
  local outdir=$1 p=$2 T=$3 type=$4 target=$5; shift 5
  local existing=$(count_ckpts "$outdir" "$p" "$T" "$type")
  [ "$existing" -ge "$target" ] && return
  for s in $(seq 1 $target); do
    if [ "$type" = "dad" ]; then
      train_dabs "$outdir" "$p" "$T" "$s" "$@"
    else
      train_static "$outdir" "$p" "$T" "$s" "$@"
    fi
    [ $(count_ckpts "$outdir" "$p" "$T" "$type") -ge "$target" ] && break
  done
}

eval_all_in() {
  local outdir=$1; shift
  for ckpt in "$outdir"/model_*.pt; do
    [ -f "$ckpt" ] || continue
    local mf="${ckpt%.pt}_eval_inter.txt"
    [ -s "$mf" ] && continue
    python3 evaluate_inter.py --model-path "$ckpt" \
      --num-rollouts 50 --seed 42 \
      --num-gibbs-samples 200 --burnin 100 "$@" > "$mf" 2>&1
  done
}

# §5.1   p=15, T in {4,8,12,16}   DABS + Static
for T in 4 8 12 16; do
  train_if_needed results_5_1_gprior        15 $T dad    3
  train_if_needed results_5_1_static_gprior 15 $T static 3
done
eval_all_in results_5_1_gprior
eval_all_in results_5_1_static_gprior

# §5.2   T=16, p in {10,20}   DABS + Static
for p in 10 20; do
  train_if_needed results_5_2_gprior        $p 16 dad    3
  train_if_needed results_5_2_static_gprior $p 16 static 3
done
eval_all_in results_5_2_gprior
eval_all_in results_5_2_static_gprior

# §5.3   p=8, T in {8,12,16}   deterministic heredity
for T in 8 12 16; do
  train_if_needed results_5_3_dethered_gprior 8 $T dad    3
  train_if_needed results_5_3_dethered_gprior 8 $T static 3
done
eval_all_in results_5_3_dethered_gprior

# §5.3   p=8, T in {8,12,16}   strong heredity, pi_int=0.5
for T in 8 12 16; do
  train_if_needed results_5_3_strongh_gprior 8 $T dad    3 --strong-heredity --pi-interaction 0.5
  train_if_needed results_5_3_strongh_gprior 8 $T static 3 --strong-heredity --pi-interaction 0.5
done
eval_all_in results_5_3_strongh_gprior --num-gibbs-samples 150 --burnin 75

# Baselines (legacy non-g)
# Myopic
for T in 4 8 12 16; do
  out="results_baselines/myopic_p15_T${T}.txt"; [ -s "$out" ] && continue
  python3 myopic_evaluate.py --num-factors 15 --num-experiments $T \
    --noise-scale 0.5 --expected-active 0.2 \
    --scale-active 1.0 --scale-inactive 0.01 \
    --num-rollouts 50 --seed 42 --output "$out" || true
done
for p in 10 20; do
  out="results_baselines/myopic_p${p}_T16.txt"; [ -s "$out" ] && continue
  python3 myopic_evaluate.py --num-factors $p --num-experiments 16 \
    --noise-scale 0.5 --expected-active 0.2 \
    --scale-active 1.0 --scale-inactive 0.01 \
    --num-rollouts 50 --seed 42 --output "$out" || true
done
for T in 8 12 16; do
  out="results_baselines/myopic_p8_T${T}.txt"; [ -s "$out" ] && continue
  python3 myopic_evaluate.py --num-factors 8 --num-experiments $T \
    --noise-scale 0.25 --expected-active 0.5 \
    --scale-active 1.0 --scale-inactive 0.01 \
    --num-rollouts 50 --seed 42 --output "$out" || true
done

# Plackett-Burman
for cfg in "15 16" "10 16" "8 12" "8 16"; do
  read p T <<< "$cfg"
  out="results_baselines/pb_p${p}_T${T}.txt"; [ -s "$out" ] && continue
  python3 eval_pb.py --num-factors $p --num-experiments $T \
    --noise-scale 0.5 --expected-active 0.2 \
    --scale-active 1.0 --scale-inactive 0.01 \
    --num-rollouts 50 --seed 42 --output "$out" || true
done

echo "DONE: $(date)"
