# HTR-VT Reproduction — Project Report

Author: Mahadi Hassan Munna · Date: 2026-10-04

This report summarizes all work completed in this project session: reproducing the paper ["HTR-VT: Handwritten Text Recognition with Vision Transformer"](https://arxiv.org/pdf/2409.08573) (Li et al., Pattern Recognition 2025) on local hardware, three original extended experiments beyond the paper, and a presentation deck built from the results. The headline technical results (tables, methods) also live in `RESULTS.md`; this report additionally covers environment setup, troubleshooting, engineering decisions, and remaining open items that `RESULTS.md` doesn't track.

---

## 1. Goal and scope

The objective, as clarified early on, was to **reproduce the paper's published results using the existing HTR-VT codebase** — not to rewrite the architecture or adapt it to new data. A secondary objective, added once the main reproduction was underway, was to go beyond the paper with a small number of original follow-up experiments, time permitting.

## 2. Environment setup

- **`apt update` SSL errors**: traced to a certificate mismatch on the ShiftKey/GitHub Desktop APT repository (their CDN serving the wrong cert for the hostname) — a server-side issue, not fixable locally. Since GitHub Desktop itself was already uninstalled, its leftover APT source file and keyring entry were removed to stop the errors at the source.
- **Conda environment (`environment.yaml`)**: the original file failed to build because of two issues:
  - `apex==0.1` was listed as a pip dependency but is dead code — a `grep` across the entire codebase confirmed it's never imported anywhere. Removed.
  - `torch==1.13.0+cu116` (and matching torchvision/torchaudio) couldn't be resolved from the default PyPI index. Fixed by adding `--extra-index-url https://download.pytorch.org/whl/cu116` as the first pip entry.
- Result: `conda env create -f environment.yaml` now succeeds cleanly, producing the `htr` environment (Python 3.8, PyTorch 1.13.0+cu116).

## 3. Dataset acquisition

All three datasets used in the paper were sourced and converted to the flat `lines/*.png|jpg` + `lines/*.txt` layout `data/dataset.py` expects:

| Dataset | Source | Notes |
|---|---|---|
| READ2016 | official release | straightforward |
| LAM | official release | straightforward |
| IAM | **substitute required** | see below |

**IAM data-source detective story**: the official FKI IAM database requires manual registration, and the FKI registration site turned out to have an expired TLS certificate (since 2026-07-29), blocking verification email delivery indefinitely. A Kaggle mirror was tried next but turned out to be word-level crops (wrong granularity for this line-level model) and was discarded. The dataset was ultimately sourced from [Teklia/IAM-line on HuggingFace](https://huggingface.co/datasets/Teklia/IAM-line) (MIT licensed) via a one-off conversion script. Its train/val/test split sizes (6482/976/2915) and character-set size (79, matching the paper's `nb_cls=80`) match the official split exactly, though exact filename-level identity with the official split couldn't be independently verified.

## 4. Training pipeline and speed optimization

- **Hardware constraint**: training ran on a single RTX 3060 (12GB) vs. the paper's RTX 4090 (24GB), forcing `--train-bs 32` instead of the paper's 128. This was validated with a smoke test (no OOM, normal loss decrease) before committing to full 100,000-iteration runs.
- **GPU utilization problem**: `nvidia-smi dmon` showed the GPU stalling at 59–87% utilization during training, bottlenecked on CPU-bound data augmentation (the paper's default `--num-workers 0`). Fixed by raising `--num-workers` to 8 for the LAM and IAM runs (READ2016 had already started before this fix and was left alone rather than restarted).
- **`cudnn.benchmark = True`** was added to `train.py`'s `main()`, a free speedup since image size is fixed at 512×64.
- **Mixed precision (AMP) was explicitly declined** — numeric fidelity to the paper's FP32 training was prioritized over speed, per explicit instruction.
- Both speed fixes are numerically inert (wall-clock only, no effect on results).
- **Unexpected system reboot**: mid-way through the LAM 75%-data run (at 83,900/100,000 iterations, 84% done), an unplanned reboot killed the process. Confirmed via `journalctl --list-boots` boot-time alignment. Since `train.py` has no checkpoint-resume support, the run had to restart from iteration 0 — a ~20-hour loss. A `watchdog_resume.sh` script plus a `cron @reboot` entry were added afterward so any future reboot automatically restarts the in-progress stage without manual intervention.

## 5. Main reproduction results

Each dataset was trained for the paper's full 100,000 iterations via `run/{read,lam,iam}.sh` (adapted for `--train-bs 32 --num-workers 8`):

| Dataset | Paper (bs=128) | Ours (bs=32) | CER Δ |
|---|---|---|---|
| READ2016 | CER 3.9% / WER 16.5% | CER 3.86% / WER 16.52% | ~0 |
| LAM | CER 2.8% / WER 7.4% | CER 3.06% / WER 8.28% | +0.26 |
| IAM | CER 4.7% / WER 14.9% | CER 5.18% / WER 16.37% | +0.48 |

READ2016 reproduces almost exactly. LAM and IAM land slightly behind, consistent in direction and magnitude with training at 1/4 the paper's batch size (SAM optimizer + weight-decay schedule were tuned for bs=128); IAM's larger gap is also plausibly influenced by the substitute data source.

## 6. Extended experiments (beyond the paper)

Three original directions were proposed, with cross-dataset generalization and robustness run first, followed by reduced-data efficiency.

### 6.1 Cross-dataset generalization (`cross_eval.py`)

Loads a source dataset's trained checkpoint and its own label converter, then runs inference directly on a *different* dataset's test images with no retraining.

| Source → Target | CER | WER | Target OOV rate |
|---|---|---|---|
| LAM → IAM | 48.84% | 88.16% | 1.09% |
| LAM → READ | 79.19% | 109.63% | 3.45% |
| IAM → LAM | 36.60% | 75.63% | 0.97% |
| IAM → READ | 78.83% | 98.43% | 3.11% |
| READ → LAM | 59.41% | 88.30% | 1.98% |
| READ → IAM | 69.35% | 94.41% | 0.97% |

**Finding**: cross-dataset CER (37–79%) is 10–20x worse than in-domain (3–5%). Vocabulary mismatch (OOV rate 1–3.5%) explains almost none of this — it's a genuine failure to transfer handwriting-style features across domains. READ2016 is the hardest target for every source (likely its distinct historical German/archival script); LAM is the easiest target but the worst source (its large, stylistically narrow training set generalizes outward worse than IAM's or READ2016's).

### 6.2 Robustness to test-time corruption (`robustness_eval.py`)

Applies Gaussian blur, rotation, and contrast adjustment to each dataset's own test images at inference time, using the already-trained checkpoints.

| Dataset | Clean | Blur σ=1/2/3 | Rotate 2°/5°/10° | Contrast 0.7/0.4/0.2 |
|---|---|---|---|---|
| IAM | 5.18% | 7.05% / 19.01% / 23.28% | 6.29% / 14.49% / 45.34% | 5.18% / 5.18% / 5.18% |
| READ2016 | 3.86% | 4.61% / 9.04% / 11.48% | 4.58% / 11.82% / 43.64% | 3.86% / 3.86% / 3.85% |
| LAM | 3.06% | 3.67% / 7.69% / 9.77% | 4.24% / 11.48% / 42.81% | 3.06% / 3.06% / 3.06% |

**Finding**: robustness tracks the training augmentation recipe exactly, not the architecture. Contrast invariance is perfect because `ColorJitter` is part of training augmentation. Blur causes up to ~4x CER degradation because the `--blur-*` CLI flags in `option.py` are dead code never wired into the actual pipeline — blur is genuinely unseen. Rotation is catastrophic (>40% CER at 10°) because training's `RandomTransform` only applies a mild perspective warp, not true rigid rotation.

### 6.3 Reduced training data / LAM data efficiency

LAM's 19,830-line training set was subsampled to 25%, 50%, and 75%, each trained for the same full 100,000 iterations to isolate the effect of unique-sample count from total compute.

**Bug hit and fixed**: a naive random 50% subsample happened to exclude every line containing the character 'K'. Since the CTC label converter's vocabulary is built only from the training list, this crashed with `KeyError` the moment validation (on the full-vocabulary val set) hit that character. Fixed with a greedy set-cover step (49 lines, covering LAM's full 89-character alphabet) force-included before random sampling fills the remaining budget. All three fractions were regenerated with this fix for consistency.

| Training data | CER | WER |
|---|---|---|
| 25% (4,957 lines) | 3.86% | 10.50% |
| 50% (9,915 lines) | 3.38% | 9.12% |
| 75% (14,872 lines) | 3.11% | 8.45% |
| 100% (19,830 lines) | 3.06% | 8.28% |

**Finding**: a clean, monotonic diminishing-returns curve (25%→50% cuts CER by 0.48, 50%→75% by 0.27, 75%→100% by just 0.05). Three-quarters of the data already gets within 0.05 CER of the full-data result.

## 7. Repository artifacts produced

- `environment.yaml` — fixed (see §2)
- `run_queue.sh` — chains READ2016 → LAM → IAM train+test end to end
- `run_reduced_data_queue.sh` — chains the three LAM data-fraction runs
- `watchdog_resume.sh` + `cron @reboot` entry — auto-resumes training after an unplanned reboot
- `cross_eval.py`, `robustness_eval.py` — the two extended-analysis scripts, reusable against any checkpoint in `output/`
- `data/LAM/train_{25,50,75}pct.ln` — coverage-guaranteed subsampled training lists
- `RESULTS.md` — full technical write-up (methods + tables) for all of the above
- `HTR-VT_Presentation.pptx` — 21-slide deck for a 20-minute talk (see §9)

## 8. Presentation deck

Built with `pptxgenjs`, iterated through three rounds (initial 12-slide build → "more professional" restyle → expansion to 21 slides for a 20-minute talk):

- **Design**: restrained navy/steel/neutral palette, no colored icon badges or decorative shapes, plain typographic numerals for enumerated items, native editable charts throughout (no screenshots).
- **Structure**: Title → Agenda → Problem/Background → How HTR-VT works (pipeline, span masking, SAM optimizer deep-dives) → Datasets & paper's results (incl. a named-competitor leaderboard table) → Our reproduction setup & engineering journey (environment bug, GPU-idle fix, mid-run reboot) → IAM data-source detective story → the three extended experiments → Limitations & Future Work → Thank You/Questions.
- **QA performed**: structural validation (`validate.py`), content/text verification (`python-pptx`, since the installed `markitdown` package turned out to be an unrelated broken package), and full visual inspection of rendered slide images — catching and fixing a blank table header, chart labels rounded to integers instead of one decimal, a stray leftover sentence accidentally left in from an earlier draft, and a dangling reference to a backup slide that was never built.
- Final file: `HTR-VT_Presentation.pptx` (539K), copied into the repo root.

## 9. Known issues and fixes applied along the way

| Issue | Fix |
|---|---|
| `apex==0.1` pip error | removed dead dependency |
| `torch==1.13.0+cu116` unresolvable | added PyTorch cu116 extra index URL |
| FKI IAM registration blocked (expired cert) | substituted Teklia HF mirror |
| Kaggle IAM mirror wrong granularity | discarded, used Teklia instead |
| GPU stalling at 59–87% utilization | `--num-workers 8` |
| Cross-eval checkpoint size mismatch (`nb_cls`) | hardcoded per-dataset `nb_cls` instead of recomputing from a rebuilt alphabet |
| Reduced-data argparse error | moved `--train-data-list` after the dataset subcommand token |
| Reduced-data `KeyError` crash | greedy set-cover character-coverage guarantee before random sampling |
| Mid-run system reboot lost ~20h of training | added cron `@reboot` watchdog, no resume-from-checkpoint existed so restarted from 0 |
| pptxgenjs table header rendering blank | set `fill` directly per header cell instead of overlaying a colored rect |
| Chart labels rounded to integers | added `dataLabelFormatCode: '0.0"%"'` |
| Accent-line/decorative-shape design violations | removed, replaced with plain typographic numerals |

## 10. Outstanding / open items

- **`.gitignore`**: the user's self-authored version was reviewed and found to have real bugs — typo'd trailing "s" on the LAM/READ2016 lines (`data/LAM/s`, `data/read2016/s` instead of `lines/`) and was missing `output/`, `__pycache__/`, and `*.log` entries entirely. **This has since been corrected** (current content: `data/iam/lines/`, `data/LAM/lines/`, `data/read2016/lines/`, `output/`, `__pycache__/`, `*.log`) — confirmed via direct read of the file, so no further action needed here.
- **Untracked files in git status** worth a decision before any commit: `2409.08573v1.pdf` (the paper itself — likely shouldn't be committed to the repo) and the LibreOffice lock file `.~lock.HTR-VT_Presentation.pptx#` (safe to delete, it's a transient lock artifact, not real content).
- **IAM substitute-data identity**: filename-level identity between the Teklia mirror and the official FKI split was never independently verified, only split sizes and character-set size. Worth flagging explicitly in any formal write-up or presentation Q&A.

---

*This report consolidates the full session. See `RESULTS.md` for the authoritative, standalone technical methods/results document, and `HTR-VT_Presentation.pptx` for the presentation built from this work.*
