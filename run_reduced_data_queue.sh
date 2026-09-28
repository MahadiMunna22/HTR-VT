#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate htr
cd /home/aihn/Documents/Github/HTR-VT

for PCT in 25 50 75; do
    echo "$(date) Starting LAM ${PCT}% training..."
    python3 train.py --exp-name lam_${PCT}pct \
    --max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 --num-workers 8 \
    --mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
    --img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
    --proba 0.5 --alpha 1 --total-iter 100000 \
    LAM --train-data-list ./data/LAM/train_${PCT}pct.ln \
    > output_lam_${PCT}pct_train.log 2>&1

    echo "$(date) LAM ${PCT}% training finished. Running test.py..."

    python3 test.py --exp-name lam_${PCT}pct \
    --max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 --num-workers 8 \
    --mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
    --img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
    --proba 0.5 --alpha 1 --total-iter 100000 \
    LAM --train-data-list ./data/LAM/train_${PCT}pct.ln \
    > output_lam_${PCT}pct_test.log 2>&1

    echo "$(date) LAM ${PCT}% test done."
done

echo "$(date) REDUCED-DATA QUEUE ALL DONE."
