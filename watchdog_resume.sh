#!/bin/bash
# Auto-resume the LAM 75% reduced-data run if the machine reboots mid-training.
# train.py has no checkpoint-resume support, so "resume" means restarting that
# stage's training from iteration 0 (not continuing from where it left off).

REPO=/home/aihn/Documents/Github/HTR-VT
MASTER_LOG=$REPO/run_reduced_data_master.log

cd "$REPO" || exit 1

if grep -q "REDUCED-DATA QUEUE ALL DONE" "$MASTER_LOG" 2>/dev/null; then
    echo "$(date) watchdog: queue already complete, nothing to do."
    exit 0
fi

if pgrep -f "train.py.*lam_75pct\|test.py.*lam_75pct" > /dev/null; then
    echo "$(date) watchdog: lam_75pct already running, nothing to do."
    exit 0
fi

echo "$(date) watchdog: resuming LAM 75% training after reboot."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate htr

python3 train.py --exp-name lam_75pct \
--max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 --num-workers 8 \
--mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
--img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
--proba 0.5 --alpha 1 --total-iter 100000 \
LAM --train-data-list ./data/LAM/train_75pct.ln \
> output_lam_75pct_train.log 2>&1

echo "$(date) LAM 75% training finished. Running test.py..." >> "$MASTER_LOG"

python3 test.py --exp-name lam_75pct \
--max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 --num-workers 8 \
--mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
--img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
--proba 0.5 --alpha 1 --total-iter 100000 \
LAM --train-data-list ./data/LAM/train_75pct.ln \
> output_lam_75pct_test.log 2>&1

echo "$(date) LAM 75% test done. REDUCED-DATA QUEUE ALL DONE." >> "$MASTER_LOG"
