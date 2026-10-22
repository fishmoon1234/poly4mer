#!/usr/bin/env bash
# Sweep the Top-K SAE over (alpha, K) with tunable LR, epochs, and LR decay.
# Optionally runs pSMILES reconstruction accuracy after each train.
#
#   alpha in {4, 8, 16, 32}                    # expansion factor (D = alpha * 768)
#   K     in {40, 80, 100, 160, 500, 1000, 2000}
#
# Usage:
#   ./sweep_topk_sae.sh
#   EPOCHS=50 LR=3e-4 LR_DECAY_MIN=0.1 ./sweep_topk_sae.sh
#   ALPHAS="8 16" KS="80 160" ./sweep_topk_sae.sh
#   DATA_PATH=data/PSMILES.smi ./sweep_topk_sae.sh
#   TAG=lr1e4 LR=1e-4 ./sweep_topk_sae.sh         # keep a tagged run separate
#   RUN_EVAL=0 ./sweep_topk_sae.sh                # skip pSMILES accuracy eval
#   FORCE=1 ./sweep_topk_sae.sh                   # retrain even if ckpt exists
#
# Each python call is prefixed with nohup so SIGHUP from a disconnected
# terminal won't kill the running train/eval. To also detach the sweep
# shell itself, launch the script with: nohup ./sweep_topk_sae.sh &
#
# Env knobs:
#   ALPHAS        list of expansion factors           (default "4 8 16 32")
#   KS            list of K values                    (default "40 80 100 160 500 1000 2000")
#   EPOCHS        training epochs         -> --epochs              (default 50)
#   LR            base learning rate      -> --lr                  (default 3e-4)
#   LR_DECAY_MIN  cosine floor frac of LR -> --lr_decay_min        (default 0.0)
#   WARMUP_STEPS  LR warmup steps         -> --warmup_steps        (default 500)
#   DATA_PATH     SMILES file             -> --data_path           (default: PSMILES.smi)
#   RUN_EVAL      1=run pSMILES acc after train                    (default 1)
#   TAG           suffix for ckpt/log/summary filenames             (default "")
#   EXTRA_ARGS    extra args passed to train_topk_sae.py
#   FORCE         1 = retrain even if ckpt exists                   (default 0)
#
# Outputs are written under $SCRIPT_DIR/checkpoint/topk_sae/ — see the
# "Results" footer at the end of the run for absolute paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/diya/anaconda3/envs/poly4mer/bin/python}"
ALPHAS="${ALPHAS:-4 8 16 32}"
KS="${KS:-40 80 100 160 500 1000 2000}"
EPOCHS="${EPOCHS:-1000}"
LR="${LR:-3e-5}"
LR_DECAY_MIN="${LR_DECAY_MIN:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data/PSMILES.smi}"
RUN_EVAL="${RUN_EVAL:-1}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE="${FORCE:-0}"

CKPT_DIR="$SCRIPT_DIR/checkpoint/topk_sae"
LOG_DIR="$CKPT_DIR/logs"
SUFFIX="${TAG:+_$TAG}"
SUMMARY="$CKPT_DIR/sweep_summary${SUFFIX}.tsv"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

if [[ ! -f "$SUMMARY" ]]; then
    printf "alpha\tk\tlr\tlr_decay_min\tepochs\tstatus\tbest_val_mse\tfraction_alive\tL0\tpsmiles_acc\tcheckpoint\ttrain_log\teval_log\n" > "$SUMMARY"
fi

echo ">> Sweep config"
echo "   alphas        : $ALPHAS"
echo "   ks            : $KS"
echo "   epochs        : $EPOCHS"
echo "   lr            : $LR"
echo "   lr_decay_min  : $LR_DECAY_MIN"
echo "   warmup_steps  : $WARMUP_STEPS"
echo "   data_path     : $DATA_PATH"
echo "   run_eval      : $RUN_EVAL"
echo "   tag           : ${TAG:-<none>}"
echo "   force         : $FORCE"
echo "   logs          -> $LOG_DIR"
echo "   summary       -> $SUMMARY"
echo

for alpha in $ALPHAS; do
    for k in $KS; do
        ckpt="$CKPT_DIR/topk_sae_alpha${alpha}_k${k}${SUFFIX}.pt"
        train_log="$LOG_DIR/alpha${alpha}_k${k}${SUFFIX}.log"
        eval_log="$LOG_DIR/alpha${alpha}_k${k}${SUFFIX}.eval.log"

        if [[ -f "$ckpt" && "$FORCE" != "1" ]]; then
            echo ">> SKIP  alpha=$alpha k=$k  (exists: $ckpt — use FORCE=1 to retrain)"
            continue
        fi

        echo ">> TRAIN alpha=$alpha k=$k lr=$LR decay_min=$LR_DECAY_MIN epochs=$EPOCHS -> $train_log"
        if ! nohup "$PYTHON" train_topk_sae.py \
                --expansion_factor "$alpha" \
                --k "$k" \
                --epochs "$EPOCHS" \
                --lr "$LR" \
                --lr_decay_min "$LR_DECAY_MIN" \
                --warmup_steps "$WARMUP_STEPS" \
                --data_path "$DATA_PATH" \
                --save_path "$ckpt" \
                $EXTRA_ARGS 2>&1 | tee "$train_log"; then
            echo ">> FAIL  alpha=$alpha k=$k  (see $train_log)"
            printf "%s\t%s\t%s\t%s\t%s\tTRAIN_FAIL\t\t\t\t\t%s\t%s\t\n" \
                "$alpha" "$k" "$LR" "$LR_DECAY_MIN" "$EPOCHS" "$ckpt" "$train_log" >> "$SUMMARY"
            continue
        fi

        best_mse=$(grep -E "Best val MSE" "$train_log" | tail -n1 \
                    | sed -E 's/.*Best val MSE: ([0-9.eE+\-]+).*/\1/' || true)
        frac_alive=$(grep -E "fraction-alive \(train\)" "$train_log" | tail -n1 \
                    | sed -E 's/.*fraction-alive \(train\): ([0-9.]+).*/\1/' || true)
        l0=$(grep -E "Best val MSE" "$train_log" | tail -n1 \
                    | sed -E 's/.*L0: ([0-9.]+).*/\1/' || true)

        psmiles_acc=""
        eval_log_val=""
        if [[ "$RUN_EVAL" == "1" ]]; then
            echo ">> EVAL  alpha=$alpha k=$k  -> $eval_log"
            if nohup "$PYTHON" eval_sae_psmiles.py \
                    --sae_ckpt "$ckpt" \
                    --data_path "$DATA_PATH" 2>&1 | tee "$eval_log"; then
                psmiles_acc=$(grep -E "pSMILES reconstruction accuracy" "$eval_log" | tail -n1 \
                              | sed -E 's/.*: ([0-9.]+)%.*/\1/' || true)
                eval_log_val="$eval_log"
            else
                echo ">> EVAL FAIL alpha=$alpha k=$k  (see $eval_log)"
                psmiles_acc="EVAL_FAIL"
                eval_log_val="$eval_log"
            fi
        fi

        printf "%s\t%s\t%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$alpha" "$k" "$LR" "$LR_DECAY_MIN" "$EPOCHS" \
            "${best_mse:-?}" "${frac_alive:-?}" "${l0:-?}" "${psmiles_acc:-?}" \
            "$ckpt" "$train_log" "$eval_log_val" >> "$SUMMARY"
    done
done

echo
echo ">> Sweep complete."
echo ">> Results"
echo "   checkpoints   : $CKPT_DIR/topk_sae_alpha*_k*${SUFFIX}.pt"
echo "   train logs    : $LOG_DIR/alpha*_k*${SUFFIX}.log"
if [[ "$RUN_EVAL" == "1" ]]; then
    echo "   eval logs     : $LOG_DIR/alpha*_k*${SUFFIX}.eval.log"
fi
echo "   summary TSV   : $SUMMARY"
echo
echo ">> Summary preview:"
column -t -s $'\t' "$SUMMARY"
