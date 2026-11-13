# HTR-VT Reproduction & Extended Analysis

Reproduction of ["HTR-VT: Handwritten Text Recognition with Vision Transformer"](https://arxiv.org/pdf/2409.08573) (Li et al., Pattern Recognition 2025), plus three additional experiments not covered in the original paper.

## 1. Setup

- **Hardware**: single NVIDIA RTX 3060 (12GB), vs. the paper's RTX 4090 (24GB)
- **Environment**: `environment.yaml` — Python 3.8, PyTorch 1.13.0+cu116
- **Key deviation**: `--train-bs 32` instead of the paper's 128, due to GPU memory constraints. This was validated first via smoke test (no OOM, normal loss decrease) before committing to full runs.
- **Speed optimizations applied** (LAM/IAM runs only, not READ2016): `--num-workers 8` (paper's default was 0, which left the GPU stalling at 59-87% utilization during CPU-bound augmentation) and `cudnn.benchmark = True` in `train.py`. Both are numerically inert — they only affect wall-clock time, not results.
- **IAM data source**: the official FKI IAM database requires manual registration; while waiting on that, IAM was instead sourced from [Teklia/IAM-line on HuggingFace](https://huggingface.co/datasets/Teklia/IAM-line) (MIT licensed). Its train/val/test split sizes (6482/976/2915) match the paper's official split exactly, and character set size (79, matching `nb_cls=80`) also matches — but exact filename-level identity with the official split could not be verified.
- All three datasets (IAM, READ2016, LAM) were converted to the flat `lines/*.png|jpg` + `lines/*.txt` layout expected by `data/dataset.py`, matching `data/format_datasets.py`'s intended output format.

## 2. Main reproduction results

Trained via `run/{read,lam,iam}.sh` (adapted for `--train-bs 32`; `--num-workers 8` for LAM and IAM only, READ2016 ran with 0), each for the paper's full 100,000 iterations.

| Dataset | Paper (bs=128) | Ours (bs=32) | CER Δ |
|---|---|---|---|
| READ2016 | CER 3.9% / WER 16.5% | CER 3.86% / WER 16.52% | ~0 |
| LAM | CER 2.8% / WER 7.4% | CER 3.06% / WER 8.28% | +0.26 |
| IAM | CER 4.7% / WER 14.9% | CER 5.18% / WER 16.37% | +0.48 |

**Take**: READ2016 reproduces almost exactly. LAM and IAM land a bit behind, consistent in direction and magnitude with training at 1/4 the paper's batch size (the SAM optimizer + weight-decay schedule were tuned for bs=128). IAM's larger gap is also plausibly influenced by the substitute data source.

Checkpoints: `output/{read,lam,iam}/best_CER.pth` (and `best_WER.pth`).

## 3. Extended analysis (beyond the paper)

### 3.1 Cross-dataset generalization

**Question**: does a model trained on one dataset transcribe another dataset's handwriting at all, with no retraining?

**Method** (`cross_eval.py`): loads a source dataset's trained checkpoint + its own CTC label converter (built from its own training charset), then runs inference directly on a *different* dataset's test images, computing CER/WER against that dataset's raw ground-truth text. Note the three datasets do **not** share an identical character set (IAM: 79 chars, plain ASCII; READ2016: 89 chars, incl. German/Latin diacritics; LAM: 89 chars, incl. Italian accents) — despite READ2016 and LAM having the same *count*, their actual character sets differ. `oov_rate` reports what fraction of the target's ground-truth characters don't even exist in the source's vocabulary, to separate "vocabulary can't represent it" from "genuine transcription failure."

| Source → Target | CER | WER | Target char OOV rate |
|---|---|---|---|
| LAM → IAM | 48.84% | 88.16% | 1.09% |
| LAM → READ | 79.19% | 109.63% | 3.45% |
| IAM → LAM | 36.60% | 75.63% | 0.97% |
| IAM → READ | 78.83% | 98.43% | 3.11% |
| READ → LAM | 59.41% | 88.30% | 1.98% |
| READ → IAM | 69.35% | 94.41% | 0.97% |
| *(in-domain reference)* | *3-5%* | *8-16%* | — |

**Finding**: cross-dataset CER (37-79%) is roughly 9-20x the target's in-domain CER (3-5%). Since OOV rate is only 1-3.5%, vocabulary mismatch explains almost none of the gap — this is genuine failure to transfer handwriting-style features across domains, not a bookkeeping artifact. READ2016 is the hardest target (~79% CER from either other model) and LAM the easiest (48.0% on average). As a source, the IAM model transfers best (57.7% mean CER on the other two datasets); the READ2016 and LAM models are equally weak (64.4% and 64.0%), so LAM's much larger training set does not buy better transfer.

### 3.2 Robustness to test-time corruption

**Question**: how does the model degrade under corruptions applied only at test time, and do corruptions covered by the training augmentation hurt less (H3)?

**Method** (`robustness_eval.py`, v2 — see the audit note below): corruptions are applied to each dataset's own test images using the trained checkpoints (no retraining). Not covered by training: Gaussian blur (σ = 1/2/3 at model resolution, kernel 2⌈3σ⌉+1), rotation (2°/5°/10°, raw line rotated with an expanded bounding box before resizing), Gaussian noise (σ = 0.05/0.1/0.2). Covered by training: stroke thickening/thinning (the training's own erosion/dilation) with kernel 2 px (training range), 3 and 4 px (beyond). Control: contrast (0.7/0.4/0.2). Full table in `output_robustness_v2.log` / `robustness_results.json`.

| CER (%) | IAM | READ2016 | LAM |
|---|---|---|---|
| clean | 5.18 | 3.87 | 3.06 |
| blur σ = 1 / 2 / 3 | 7.13 / 54.79 / 88.87 | 4.69 / 35.87 / 91.01 | 3.70 / 18.62 / 63.92 |
| rotate 2° / 5° / 10° | 7.32 / 17.79 / 60.67 | 4.28 / 6.48 / 20.44 | 4.63 / 13.62 / 65.95 |
| noise 0.05 / 0.1 / 0.2 | 13.43 / 45.70 / 91.10 | 6.28 / 36.93 / 83.64 | 7.06 / 25.18 / 67.90 |
| thicken 2 / 3 / 4 px | 5.72 / 8.54 / 18.53 | 4.07 / 4.82 / 7.36 | 3.30 / 4.74 / 12.59 |
| thin 2 / 3 / 4 px | 5.81 / 17.22 / 60.07 | 4.01 / 5.42 / 17.80 | 3.28 / 6.86 / 30.82 |
| contrast 0.7 / 0.4 / 0.2 | 5.18 / 5.18 / 5.18 | 3.86 / 3.86 / 3.85 | 3.06 / 3.06 / 3.06 |

**Finding**: H3 holds. Covered corruptions at training strength (2 px) cost only +0.14 to +0.63 CER points; every uncovered corruption costs more even at its mildest level (+0.4 to +8.3), and moderate levels are destructive (blur σ = 2: 18.6–54.8%; noise σ = 0.1: 25.2–45.7%; rotation 10°: 20.4–66.0%). Coverage only protects the trained range: thinning by 4 px reaches 60.1% CER on IAM. READ2016 is much more robust to rotation because its lines are short (median 21 characters), so the tilted bounding box barely grows and glyphs shrink less at the fixed 64-px input height.

**Contrast is not learned robustness.** The model applies a parameter-free layer norm to every input image (`model/HTR_VT.py`, first line of `forward`), i.e. per-image standardization. torchvision's contrast reduction is an affine change of each image, which this removes exactly — hence identical CER at every level. (The earlier version of this document credited `ColorJitter`; that was wrong.)

**Audit note — v1 of this experiment had three problems, now fixed**: (1) a fixed 5-px blur kernel truncated σ = 2 and 3 into nearly the same box filter, understating blur damage (IAM σ = 2: 19.0% in v1 vs 54.8% now); (2) rotation was applied to the padded 512×64 canvas, cutting off the line ends (30.1% of ink lost at 10° over the IAM test set; reproduce with `report/audit_v1_checks.py`) — kept as `rotate_canvas` for comparison (42.8–45.3% at 10° for all datasets); (3) no corruption in v1 was actually covered by training, so H3 could not be tested — stroke thickening/thinning and noise were added.

### 3.3 Reduced training data (LAM data efficiency)

**Question**: how much does performance degrade as unique training data becomes scarcer?

**Method**: LAM's 19,830-line training set was randomly subsampled (seed=42, `data/LAM/train_{25,50,75}pct.ln`) to 25% (4,957 lines), 50% (9,915 lines), and 75% (14,872 lines), holding val/test fixed. Each fraction trains for the **same 100,000 iterations** as the full-data run (`run_reduced_data_queue.sh`), so this isolates the effect of unique-sample count rather than confounding it with less total training compute — the model simply repeats the smaller pool more often.

**Gotcha hit and fixed**: a naive random subsample can, by chance, exclude every training line containing some character (here, a plain random 50% draw happened to drop every line with the letter 'K'). Since the CTC label converter's vocabulary is built only from the *training* list, this crashes with `KeyError` the moment validation (on the untouched, full-vocabulary val set) encounters that character. The fix: for each fraction, a small greedy set-cover (49 lines, out of LAM's full 89-character alphabet, sufficed) is force-included to guarantee full character coverage regardless of the random draw, with the remaining budget filled by uniform random sampling. All reported fractions below use this coverage-guaranteed sampling.

| Training data | CER | WER |
|---|---|---|
| 25% (4,957 lines) | 3.86% | 10.50% |
| 50% (9,915 lines) | 3.38% | 9.12% |
| 75% (14,872 lines) | 3.11% | 8.45% |
| 100% (19,830 lines, §2 above) | 3.06% | 8.28% |

**Finding**: a clean, monotonic diminishing-returns curve. Going from 25%→50% of the data cuts CER by 0.48 points; 50%→75% only by 0.27; 75%→100% by just 0.05. Three-quarters of LAM's training data already gets to within 0.05 CER of the full-data result — the model captures most of what's learnable from a comparatively small fraction of the training set, consistent with the paper's own claim of data efficiency, and extending that claim to a dimension the paper itself never tested.

## 4. Reproducibility

All scripts referenced above live in the repo root / `data/`:
- `environment.yaml` — fixed conda env (removed dead `apex==0.1` dep, added PyTorch cu116 index)
- `run_queue.sh` — chains READ2016 → LAM → IAM train+test end to end
- `run_reduced_data_queue.sh` — chains the three LAM data-fraction runs
- `cross_eval.py`, `robustness_eval.py` — the two extended analyses above, reusable against any checkpoint in `output/`
- `report/make_figures.py` + `report/report.typ` — the full Typst report (figures, side-by-side test examples, error analysis, bootstrap intervals, audit)
