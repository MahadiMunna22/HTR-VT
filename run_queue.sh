#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate htr
cd /home/aihn/Documents/Github/HTR-VT

READ_PID=590630

echo "$(date) Waiting for READ2016 training (PID $READ_PID) to finish..."
while kill -0 "$READ_PID" 2>/dev/null; do
    sleep 30
done
echo "$(date) READ2016 training finished. Running test.py for READ..."

python3 test.py --exp-name read \
--max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 \
--mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
--img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
--proba 0.5 --alpha 1 --total-iter 100000 \
READ > output_read_test.log 2>&1

echo "$(date) READ2016 test done. Starting LAM training..."

python3 train.py --exp-name lam \
--max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 --num-workers 8 \
--mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
--img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
--proba 0.5 --alpha 1 --total-iter 100000 \
LAM > output_lam_train.log 2>&1

echo "$(date) LAM training finished. Running test.py for LAM..."

python3 test.py --exp-name lam \
--max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 --num-workers 8 \
--mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
--img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
--proba 0.5 --alpha 1 --total-iter 100000 \
LAM > output_lam_test.log 2>&1

echo "$(date) LAM test done. Starting IAM training..."

python3 train.py --exp-name iam \
--max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 --num-workers 8 \
--mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
--img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
--proba 0.5 --alpha 1 --total-iter 100000 \
IAM > output_iam_train.log 2>&1

echo "$(date) IAM training finished. Running test.py for IAM..."

python3 test.py --exp-name iam \
--max-lr 1e-3 --train-bs 32 --val-bs 8 --weight-decay 0.5 --num-workers 8 \
--mask-ratio 0.4 --attn-mask-ratio 0.1 --max-span-length 8 \
--img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 \
--proba 0.5 --alpha 1 --total-iter 100000 \
IAM > output_iam_test.log 2>&1

echo "$(date) ALL DONE."
