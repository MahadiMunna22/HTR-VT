"""
Reproduce the two numbers the audit (report Section 6) quotes for the flawed first
version of the robustness experiment, whose log no longer exists:
  1. share of ink pixels cut off when the padded 64x512 canvas is rotated (v1 rotation)
  2. IAM CER under blur with the v1 fixed 5x5 kernel versus the corrected kernel

Run from the repo root:  python report/audit_v1_checks.py
Writes report/figures/audit_v1.json.
"""
import json
import math
import os
import sys

import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.getcwd())
import robustness_eval as R
from data import dataset

fns_ds = dataset.myLoadDS('./data/iam/test.ln', './data/iam/lines/', R.IMG_SIZE)
x = torch.stack([torch.from_numpy(fns_ds[i][0]) for i in range(len(fns_ds))]).float()
ink = (x < 0.5).float().sum().item()
ink_kept = {a: (TF.rotate(x, angle=a, fill=1.0) < 0.5).float().sum().item() / ink for a in [2, 5, 10]}

R.CORRUPTIONS['blur_v1'] = ('tensor', lambda t, s: TF.gaussian_blur(t, kernel_size=5, sigma=s))
model, converter = R.load_model_and_converter('IAM')
blur = {}
for s in [1.0, 2.0, 3.0]:
    v1, _ = R.evaluate('IAM', model, converter, 'blur_v1', s)
    v2, _ = R.evaluate('IAM', model, converter, 'blur', s)
    blur[str(s)] = dict(v1_kernel=5, v1_cer=100 * v1, v2_kernel=2 * math.ceil(3 * s) + 1, v2_cer=100 * v2)

out = dict(n_iam_test_lines=len(fns_ds),
           canvas_rotation_ink_lost_pct={str(a): 100 * (1 - k) for a, k in ink_kept.items()},
           iam_blur=blur)
json.dump(out, open('report/figures/audit_v1.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
