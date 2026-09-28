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

Trained via `run/{read,lam,iam}.sh` (adapted for `--train-bs 32`, `--num-workers 8`), each for the paper's full 100,000 iterations.

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

**Finding**: cross-dataset CER (37-79%) is 10-20x worse than in-domain (3-5%). Since OOV rate is only 1-3.5%, vocabulary mismatch explains almost none of the gap — this is genuine failure to transfer handwriting-style features across domains, not a bookkeeping artifact. Two consistent asymmetries: **READ2016 is the hardest target for everyone** (~79% CER regardless of source, plausibly due to its distinct historical German/archival script), and **LAM is the easiest target but the worst source** — its large, stylistically narrow training set doesn't produce features that generalize outward as well as IAM's or READ2016's do.

### 3.2 Robustness to test-time corruption

**Question**: how does the model degrade under corruptions never seen during training?

**Method** (`robustness_eval.py`): applies blur (Gaussian, σ = 1/2/3), rotation (2°/5°/10°), and contrast adjustment (factor 0.7/0.4/0.2) to each dataset's own in-domain test images at inference time, using the existing trained checkpoints (no retraining).

| Dataset | Clean | Blur σ=1/2/3 | Rotate 2°/5°/10° | Contrast 0.7/0.4/0.2 |
|---|---|---|---|---|
| IAM | 5.18% | 7.05% / 19.01% / 23.28% | 6.29% / 14.49% / 45.34% | 5.18% / 5.18% / 5.18% |
| READ2016 | 3.86% | 4.61% / 9.04% / 11.48% | 4.58% / 11.82% / 43.64% | 3.86% / 3.86% / 3.85% |
| LAM | 3.06% | 3.67% / 7.69% / 9.77% | 4.24% / 11.48% / 42.81% | 3.06% / 3.06% / 3.06% |

(CER shown; see script output for WER.)

**Finding**: robustness tracks the training augmentation recipe exactly, not the architecture.
- **Contrast**: perfectly invariant, because `ColorJitter` (brightness/contrast/saturation/hue) is applied during training (`data/dataset.py`'s `SameTrCollate`).
- **Blur**: causes up to ~4x CER degradation. The `--blur-*` CLI flags in `utils/option.py` are dead code — never wired into the actual augmentation pipeline (`data/transform.py` has no blur class at all) — so blur is genuinely unseen.
- **Rotation**: catastrophic (>40% CER at just 10°). Training's `RandomTransform` only applies a mild ±8px corner-perspective warp, not true rigid rotation — a qualitatively different and far harsher transform, especially on these wide 512×64 line images where rotation causes large vertical displacement at the line's ends.

### 3.3 Reduced training data (LAM data efficiency)

**Question**: how much does performance degrade as unique training data becomes scarcer?

**Method**: LAM's 19,830-line training set was randomly subsampled (seed=42, `data/LAM/train_{25,50,75}pct.ln`) to 25% (4,957 lines), 50% (9,915 lines), and 75% (14,872 lines), holding val/test fixed. Each fraction trains for the **same 100,000 iterations** as the full-data run (`run_reduced_data_queue.sh`), so this isolates the effect of unique-sample count rather than confounding it with less total training compute — the model simply repeats the smaller pool more often.

**Gotcha hit and fixed**: a naive random subsample can, by chance, exclude every training line containing some character (here, a plain random 50% draw happened to drop every line with the letter 'K'). Since the CTC label converter's vocabulary is built only from the *training* list, this crashes with `KeyError` the moment validation (on the untouched, full-vocabulary val set) encounters that character. The fix: for each fraction, a small greedy set-cover (49 lines, out of LAM's full 89-character alphabet, sufficed) is force-included to guarantee full character coverage regardless of the random draw, with the remaining budget filled by uniform random sampling. All reported fractions below use this coverage-guaranteed sampling.

| Training data | CER | WER |
|---|---|---|
| 25% (4,957 lines) | *pending* | *pending* |
| 50% (9,915 lines) | *pending* | *pending* |
| 75% (14,872 lines) | *pending* | *pending* |
| 100% (19,830 lines, §2 above) | 3.06% | 8.28% |

*(Status: running as of 2026-09-21; results to be filled in once complete.)*

## 4. Reproducibility

All scripts referenced above live in the repo root / `data/`:
- `environment.yaml` — fixed conda env (removed dead `apex==0.1` dep, added PyTorch cu116 index)
- `run_queue.sh` — chains READ2016 → LAM → IAM train+test end to end
- `run_reduced_data_queue.sh` — chains the three LAM data-fraction runs
- `cross_eval.py`, `robustness_eval.py` — the two extended analyses above, reusable against any checkpoint in `output/`
