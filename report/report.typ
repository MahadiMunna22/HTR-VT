// HTR-VT reproduction and stress-test report.
// Build: python report/make_figures.py && typst compile report/report.typ

#let M = json("figures/metrics.json")
#let E = json("figures/examples.json")
#let A = json("figures/error_analysis.json")
#let B = json("figures/bootstrap.json")
#let V1 = json("figures/audit_v1.json")

#let navy = rgb("#1E2761")
#let muted = rgb("#5A5A66")
#let bad = rgb("#B42318")
#let badbg = rgb("#FBE3E1")
#let okc = rgb("#1B7F3B")

#set document(title: "Reproducing and Stress-Testing HTR-VT", author: "Md. Mahadi Hassan Munna")
#set page(paper: "a4", margin: (x: 2.2cm, top: 2.3cm, bottom: 2.3cm),
  numbering: "1",
  header: context if counter(page).get().first() > 1 [
    #set text(8pt, fill: muted)
    Reproducing and Stress-Testing HTR-VT #h(1fr) Md. Mahadi Hassan Munna
  ])
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em, spacing: 1.0em)
#set heading(numbering: "1.1")
#show heading: set text(fill: navy)
#show heading.where(level: 1): it => { v(0.6em); it; v(0.25em) }
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(9pt)
#show link: set text(fill: navy)
#set table(stroke: none, inset: (x: 5pt, y: 3.5pt))

// ---------------------------------------------------------------- helpers

#let fmt(x, d: 2) = {
  let s = str(calc.round(float(x), digits: d))
  if d == 0 { return s.split(".").at(0) }
  if not s.contains(".") { s += "." }
  s + "0" * (d - s.split(".").at(1).len())
}
#let pc(x, d: 2) = fmt(x, d: d) + "%"
#let thou(n) = {
  let s = str(int(n))
  let out = ""
  for (i, ch) in s.clusters().rev().enumerate() {
    if i > 0 and calc.rem(i, 3) == 0 { out = "," + out }
    out = ch + out
  }
  out
}
#let th(..cells) = table.header(..cells.pos().map(c => strong(c)))
#let sev(ds, c, s) = M.robustness.at(ds).at(c).at(s).cer
#let mono(body) = text(font: "DejaVu Sans Mono", size: 7.6pt, body)
#let vis(c) = if c == " " { "␣" } else { c }

#let mark-gt(runs) = if runs.len() == 0 [—] else {
  runs.map(r => if r.at(1) { highlight(fill: badbg, extent: 0.5pt, r.at(0)) } else { r.at(0) }).join()
}
#let mark-pred(runs) = if runs.len() == 0 [#text(fill: bad)[(empty output)]] else {
  runs.map(r => if r.at(1) { text(fill: bad, weight: "bold", r.at(0)) } else { r.at(0) }).join()
}
#let lineimg(path, h: 1.15cm) = box(width: 100%, height: h, image(path, width: 100%, height: 100%, fit: "contain"))

#let transcription(r, tag: none) = {
  set par(justify: false, leading: 0.45em)
  if tag != none { text(8pt, weight: "bold", fill: navy, tag); h(0.5em) }
  text(8pt, fill: muted)[CER #pc(r.cer, d: 1)]
  linebreak()
  text(7.5pt, fill: muted)[GT#h(0.9em)]; mono(mark-gt(r.gt_runs))
  linebreak()
  text(7.5pt, fill: muted)[Pred#h(0.25em)]; mono(mark-pred(r.pred_runs))
}

#let side-by-side(rows, tag: r => none, img: r => r.img, h: 1.15cm) = table(
  columns: (50%, 1fr), inset: (x: 4pt, y: 5pt), align: (left + horizon, left + horizon),
  stroke: (x, y) => (bottom: 0.4pt + rgb("#DCDCE2")),
  ..rows.map(r => (lineimg(img(r), h: h), transcription(r, tag: tag(r)))).flatten()
)

#let callout(title, body) = block(
  width: 100%, fill: rgb("#F2F4FA"), inset: 9pt, radius: 3pt, stroke: 0.5pt + rgb("#C9D1E8"),
  [#text(weight: "bold", fill: navy, title) \ #body])

// ---------------------------------------------------------------- title

#align(center)[
  #text(18pt, weight: "bold", fill: navy)[Reproducing and Stress-Testing HTR-VT]
  #v(-0.3em)
  #text(12pt)[Data Scarcity, Domain Shift, and Corruption in Vision-Transformer Handwriting Recognition]
  #v(0.1em)
  #text(10pt, fill: muted)[Project Report · Pattern Recognition Group, University of Bern]
]
#v(0.4em)
#grid(columns: (auto, 1fr), column-gutter: 1.5em, row-gutter: 0.45em,
  text(fill: muted)[Student], [Md. Mahadi Hassan Munna (#link("mailto:mahadi.munna@students.unibe.ch")[mahadi.munna\@students.unibe.ch]), matr. 25-118-035],
  text(fill: muted)[Supervisor], [Kaspar Riesen],
  text(fill: muted)[Task category], [Task 7: Reproduce, Then Break It (robustness part also Task 5)],
  text(fill: muted)[Paper], [Y. Li, D. Chen, T. Tang, X. Shen. _HTR-VT: Handwritten Text Recognition with Vision Transformer._ Pattern Recognition, 2025 (arXiv:2409.08573)],
)
#v(0.3em)
#line(length: 100%, stroke: 0.5pt + rgb("#DCDCE2"))

// ---------------------------------------------------------------- summary

#callout[Summary][
This report documents a complete reproduction of HTR-VT on IAM, READ2016 and LAM using the authors' code on a single RTX 3060 (12 GB), followed by three stress tests that the paper does not report: training-set reduction, cross-dataset transfer and test-time corruption. All four research questions of the proposal have been answered; Section 6 lists the bugs found while auditing the experiments and how they were fixed.

#table(columns: (auto, 1fr), inset: (x: 3pt, y: 3pt),
  [*RQ1* Reproduction], [All three datasets reproduce within the 1-point target: IAM #pc(M.main.IAM.cer) (paper 4.7%), READ2016 #pc(M.main.READ.cer) (3.9%), LAM #pc(M.main.LAM.cer) (2.8%) CER, at a quarter of the paper's batch size.],
  [*RQ2 / H1* Less data], [Confirmed. LAM CER falls monotonically from #pc(M.reduced.at(0).cer) (25%) to #pc(M.reduced.at(3).cer) (100%) with strongly diminishing returns; the 75% → 100% step is within test-sample noise.],
  [*RQ3 / H2* Transfer], [Confirmed. Cross-dataset CER is 37–79%, roughly 9–20× the in-domain error of the target dataset, while only 1–3.5% of target characters are missing from the source vocabulary.],
  [*RQ4 / H3* Corruption], [Confirmed after correcting the experiment: stroke-width changes of the strength used in training cost at most 0.6 points, while blur, rotation and noise — never seen in training — cost more even at their mildest level and push CER to 18–66% at moderate levels (up to 91% at the strongest). The protection does not extend beyond the trained range. Contrast is a special case: the model is invariant by construction (to within 0.02 points).],
)
]

// ---------------------------------------------------------------- 1

= Introduction

Handwritten text recognition (HTR) turns images of handwriting into text. HTR-VT asks whether a Vision Transformer (ViT), usually data-hungry, can compete on the comparatively small public HTR datasets *without* pre-training or synthetic data. It reports state-of-the-art character error rates (CER) of 4.7% on IAM, 3.9% on READ2016 and 2.8% on LAM.

The seminar task "Reproduce, Then Break It" has two halves, and this report follows them in order:

+ *Reproduce* (Section 3): rebuild the environment, obtain the data, train the model with the paper's protocol on weaker hardware, and compare against the published numbers, including the test images and the model's transcriptions.
+ *Break* (Section 4): probe the claim along three axes the paper does not cover — how much training data is really needed (RQ2), whether the model transfers to another dataset (RQ3), and how it degrades under image corruptions (RQ4).

Section 5 looks one level below aggregate CER at the kinds of errors the models make, Section 6 is an audit of every experiment for bugs, and Section 7 lists the limitations.

// ---------------------------------------------------------------- 2

= The HTR-VT model in brief

HTR-VT reads a single pre-segmented text line, resized to 64 × 512 pixels, in five stages:

+ *Input standardization.* Each image is normalized to zero mean and unit variance (a parameter-free layer norm). This detail turns out to matter for the robustness results (Section 4.3).
+ *CNN feature extractor.* A modified ResNet-18 turns the image into a sequence of 128 feature vectors ("tokens"), each describing a narrow vertical slice of the line. The CNN supplies the local stroke structure a plain ViT would have to learn from scratch.
+ *Transformer encoder.* Four transformer layers (768 dimensions, 6 heads, sinusoidal position embeddings) let every slice attend to the whole line.
+ *CTC decoding.* A linear layer predicts a character (or "blank") per token; Connectionist Temporal Classification collapses repeats and blanks into the final text, so no character-level alignment is needed for training.
+ *Two regularizers for small data.* *Span masking* replaces contiguous runs of tokens (ratio 0.4, span length up to 8) by a learned mask token during training, forcing the model to infer missing parts of a word from context. *Sharpness-Aware Minimization* (SAM) steers training toward flat minima of the loss, which generalize better.

#figure(
  table(columns: 4, align: (left, left, left, left),
    table.hline(),
    th([Setting], [Paper], [Setting], [Paper]),
    table.hline(stroke: 0.4pt),
    [Image size], [64 × 512 px], [Max. learning rate], [10#super[−3]],
    [CNN front end], [ResNet-18 (modified)], [Weight decay], [0.5],
    [Encoder layers / heads], [4 / 6], [LR schedule], [1,000 warm-up, cosine],
    [Embedding / FFN size], [768 / 3,072], [Training budget], [100,000 iterations],
    [Optimizer], [AdamW + SAM], [Batch size], [*128* (ours: *32*)],
    [Span masking], [ratio 0.4, span 8], [EMA decay], [0.9999],
    table.hline(),
  ),
  caption: [Reference training configuration of HTR-VT. The only change in the learning setup is the batch size.],
) <tab:config>

// ---------------------------------------------------------------- 3

= Reproduction, step by step

== Step 1 — Environment

The repository's `environment.yaml` did not build. Two fixes were needed: `apex==0.1` was removed (it is never imported anywhere in the code), and the PyTorch CUDA 11.6 wheel index was added so that `torch==1.13.0+cu116` resolves. The resulting environment is Python 3.8, PyTorch 1.13.0, torchvision 0.14.0, running on an NVIDIA RTX 3060 with 12 GB of memory (the paper used an RTX 4090 with 24 GB).

== Step 2 — Data

All three datasets were converted to the flat layout the code expects (`lines/<id>.png` plus `lines/<id>.txt`, and `train/val/test.ln` file lists).

- *READ2016* and *LAM* were taken from their official releases.
- *IAM* requires registration with the FKI server, whose TLS certificate had expired, blocking account verification. A Kaggle mirror turned out to contain word crops rather than lines and was discarded. IAM was finally taken from the MIT-licensed `Teklia/IAM-line` mirror on HuggingFace. Its split sizes (6,482 / 976 / 2,915) and its 79-character alphabet match the paper's split exactly; file-level identity with the official split could not be checked (Section 7).

#figure(
  table(columns: 6, align: (left, right, right, right, right, left),
    table.hline(),
    th([Dataset], [Train], [Val], [Test], [Classes], [Content]),
    table.hline(stroke: 0.4pt),
    [IAM], [6,482], [976], [2,915], [80], [modern English, 657 writers],
    [READ2016], [8,349], [1,040], [1,138], [90], [Early Modern German council records],
    [LAM], [19,830], [2,470], [3,523], [90], [Italian letters, one author over 60 years],
    table.hline(),
  ),
  caption: [Datasets as used. "Classes" includes the CTC blank. Integrity checks (Section 6): no image file appears in two splits.],
) <tab:data>

== Step 3 — Training protocol

The batch size had to drop from 128 to 32 to fit in 12 GB. A short smoke run confirmed stable, decreasing loss before the full runs were committed, and the iteration budget was kept at 100,000, so each run sees a quarter of the paper's samples. Mixed precision was deliberately *not* used so the numerics stay comparable with the reference.

Two changes that affect only speed, not results, were made after the first run had started: data-loading workers were raised from 0 to 8 (monitoring showed the GPU idling at 59–87% utilization while waiting for CPU-side augmentation), and `cudnn.benchmark` was enabled. READ2016 therefore trained for about 30 hours, LAM and IAM for about 24.5 hours each (the paper reports about 16 hours on its RTX 4090). A system reboot killed one later run (LAM 75%) at iteration 83,900; because the code cannot resume from a checkpoint it was restarted from zero, and a cron `@reboot` watchdog was added to restart automatically in future.

The checkpoint with the lowest *validation* CER (evaluated every 1,000 iterations) is the one tested; the test set is touched once per model. @fig:curves shows the validation curves: all runs converge smoothly and are still improving slowly at the end, as expected from the cosine schedule.

#figure(image("figures/val_curves.svg", width: 100%),
  caption: [Validation CER during training. Left: the three main runs. Right: the reduced-data runs on LAM (Section 4.1).]) <fig:curves>

== Step 4 — Results

#figure(
  table(columns: 7, align: (left, right, right, right, right, right, right),
    table.hline(),
    th([Dataset], [Paper CER], [Our CER], [95% CI], [Δ CER], [Paper WER], [Our WER]),
    table.hline(stroke: 0.4pt),
    ..("IAM", "READ", "LAM").map(d => {
      let m = M.main.at(d)
      let delta = m.cer - m.paper_cer
      ([#if d == "READ" [READ2016] else [#d]], pc(m.paper_cer, d: 1), [*#pc(m.cer)*],
       [#fmt(B.main.at(d).at(0))–#fmt(B.main.at(d).at(1))],
       [#if delta >= 0 [+] else [−]#fmt(calc.abs(delta))], pc(m.paper_wer, d: 1), pc(m.wer))
    }).flatten(),
    table.hline(),
  ),
  caption: [Reproduction results on the test sets. 95% confidence intervals come from bootstrapping the test lines (2,000 resamples) and reflect test-sample variation only; run-to-run training variation could not be measured (one run per configuration).],
) <tab:main>

#figure(image("figures/reproduction.svg", width: 92%), caption: [Paper versus reproduction.]) <fig:repro>

READ2016 reproduces almost exactly. LAM (+#fmt(M.main.LAM.cer - 2.8) points) and IAM (+#fmt(M.main.IAM.cer - 4.7) points) land slightly behind, in the direction expected from training with a quarter of the batch size: SAM's neighbourhood size and the strong weight decay were tuned for batch 128, and the model sees four times fewer samples in the same number of iterations. For IAM the substitute data source may contribute as well. Every deviation is inside the proposal's target of one percentage point, so *RQ1 is answered positively*: the published claim holds on consumer hardware.

== Step 5 — Test images and transcriptions

Aggregate CER hides what the model actually outputs. The tables below show, for each dataset, four test lines chosen by a fixed rule: an exactly correct line, the median line among those with at least one error ("typical error"), the line at the 90th error percentile, and the single worst line. In the ground truth (GT), characters the model got wrong are #highlight(fill: badbg)[highlighted]; in the prediction, wrong or inserted characters are #text(fill: bad, weight: "bold")[red].

#for (ds, name) in (("IAM", "IAM"), ("READ", "READ2016"), ("LAM", "LAM")) {
  figure(
    side-by-side(E.qualitative.at(ds), tag: r => r.label),
    caption: [#name test lines and the #name model's transcriptions. #thou(A.at(ds).n_lines) test lines, of which #pc(A.at(ds).exact_lines, d: 1) are transcribed without any error.],
  )
}

Two patterns stand out. Typical errors are one or two characters inside otherwise correct lines — a confusable letter pair or a dropped short word — which is why WER is 2.7–4.3 times CER: one wrong character makes a whole word wrong. And in all three datasets the single worst line is not ordinary handwriting: an IAM "line" that is actually the "Sentence Database" title of the IAM form, wrongly segmented as a text line and labelled as a row of dashes, a three-digit page number in READ2016 ("226" read as "336"), where each misread digit costs 33 points of CER, and in LAM a struck-out word that the annotation replaces by `#` but the model reads anyway (Section 5).

// ---------------------------------------------------------------- 4

= Breaking it: three stress tests

== Less training data (RQ2, H1)

*Setup.* LAM's 19,830 training lines were subsampled to 25%, 50% and 75% with a fixed seed; validation and test sets stay unchanged. Every fraction is trained for the same 100,000 iterations, so the experiment changes the number of *distinct* lines, not the amount of training; smaller subsets are simply repeated more often (@tab:reduced).

*A bug found on the way.* The first, purely random 50% subset happened to contain no line with a capital "K". Because the label vocabulary is built from the training list, validation crashed with `KeyError: 'K'` the first time it met that letter. All subsets were therefore regenerated: a greedy set cover first picks 49 lines that together contain all 89 LAM characters, and the rest of the budget is filled at random. Every subset now has exactly the full-data alphabet, so the label-to-index mapping — and hence the output layer — is identical across all four models.

#figure(
  table(columns: 7, align: (left, right, right, right, right, right, right),
    table.hline(),
    th([Training data], [Lines], [Epochs], [Best val CER], [Test CER], [95% CI], [Test WER]),
    table.hline(stroke: 0.4pt),
    ..M.reduced.map(r => (
      [#r.pct%], [#thou(r.n)], [#fmt(r.epochs, d: 0)], pc(r.best_val_cer), [*#pc(r.cer)*],
      [#fmt(B.reduced.at(str(r.pct)).at(0))–#fmt(B.reduced.at(str(r.pct)).at(1))], pc(r.wer),
    )).flatten(),
    table.hline(),
  ),
  caption: [Reduced-data results on LAM. Epochs = 100,000 iterations × 32 / lines.],
) <tab:reduced>

#figure(image("figures/reduced_data.svg", width: 62%), caption: [Test error versus share of LAM training lines.]) <fig:reduced>

#let rd = B.reduced_diff
*Result.* CER falls monotonically, and each additional quarter of data buys less than the previous one: #fmt(rd.at(0).mean) points (25% → 50%), #fmt(rd.at(1).mean) points (50% → 75%) and #fmt(rd.at(2).mean) points (75% → 100%). A paired bootstrap over test lines puts the last gain at #fmt(rd.at(2).lo) to #fmt(rd.at(2).hi) points (95% interval)#if rd.at(2).lo <= 0 [, an interval that includes zero: on this test set the 75% and 100% models cannot be distinguished] else [, small but distinguishable from zero]. The first two gains are clearly significant (intervals #fmt(rd.at(0).lo)–#fmt(rd.at(0).hi) and #fmt(rd.at(1).lo)–#fmt(rd.at(1).hi)). *H1 is confirmed.* To put the numbers in context with the paper's LAM table: the 25% model (about 5,000 lines, #pc(M.reduced.at(0).cer)) already beats ViT (6.1%), DeiT (5.9%), GFCN (5.2%) and CNN+BLSTM (5.8%) trained on all the data, and the 50% model (#pc(M.reduced.at(1).cer)) also beats CRNN (3.8%) and TrOCR (3.6%, which uses extra training data). This supports the paper's data-efficiency claim in a dimension the paper itself never tested.

#let red = E.reduced
*What the gain looks like on individual lines.* The figures below follow three LAM test lines through all four models. They were chosen by a fixed rule: the number of character errors must drop at *every* step, and the 100% model must read the line perfectly. Lines like this are rare — only #red.n_strict of #thou(red.n_lines) test lines improve at every step — because most lines are either already correct with 25% of the data or stay difficult for all four models. Comparing just the 25% and 100% models, #thou(red.n_better) lines get better, #thou(red.n_worse) get worse and #thou(red.n_same) are unchanged; the average gain comes from many small, sometimes cancelling changes. Share of test lines transcribed perfectly: #("25", "50", "75", "100").map(p => [#p%: #pc(red.exact_lines.at(p), d: 1)]).join(", ").

#for (k, ex) in red.lines.enumerate() {
  figure(
    side-by-side(ex.rows.map(r => r + (img: ex.img)), tag: r => [#r.pct% data]),
    caption: [LAM test line #(k + 1) read by the four data-fraction models. Character errors: #ex.errors.map(e => str(e)).join(" → ") (25% → 50% → 75% → 100%).],
  )
}

== Transfer to another dataset (RQ3, H2)

*Setup.* Each of the three trained models is applied, without retraining, to the test sets of the other two datasets (six pairs). Decoding uses the source model's own vocabulary; CER and WER are computed against the target's original transcriptions. Because the alphabets differ, the *out-of-vocabulary (OOV) rate* — the share of target characters the source model cannot output at all — is reported too: it is a hard lower bound on how much of the error is caused by the vocabulary rather than by reading the handwriting.

#let names = ("IAM", "READ", "LAM")
#let full = (IAM: "IAM", READ: "READ2016", LAM: "LAM")
#figure(
  table(columns: 5, align: (left, left, right, right, right),
    table.hline(),
    th([Trained on], [Tested on], [CER], [WER], [OOV rate]),
    table.hline(stroke: 0.4pt),
    ..names.map(s => names.filter(t => t != s).map(t => {
      let c = M.cross.at(s).at(t)
      (full.at(s), full.at(t), pc(c.cer), pc(c.wer), pc(c.oov))
    })).flatten(),
    table.hline(),
  ),
  caption: [Cross-dataset results. In-domain CER is 3–5% for comparison. WER can exceed 100% because inserted words also count as errors.],
) <tab:cross>

#figure(image("figures/cross_heatmap.svg", width: 50%), caption: [CER (%) for every source/target pair; the diagonal is in-domain.]) <fig:cross>

*Result.* Transfer fails across the board: 37–79% CER instead of 3–5%. The OOV rate (1–3.5%) explains only a small fraction of that gap, so the failure is genuinely about handwriting style, not bookkeeping. *H2 is confirmed.* READ2016 (Early Modern German council records) is the hardest target for both other models, around 79% CER, and LAM the easiest (48.0% on average). As a source, the IAM model transfers best (57.7% mean CER on the other two datasets); the READ2016 and LAM models are equally weak (64.4% and 64.0%), so the much larger LAM training set does not buy better transfer. The examples below show what "transfer" looks like in practice: the model still produces plausible character strings in its own training language, which is why WER sits near 100%.

#for t in names {
  let ex = E.cross.at(t)
  figure(
    side-by-side(ex.rows.map(r => r + (img: ex.img)), tag: r => [#full.at(r.source) model]),
    caption: [A #full.at(t) test line read by all three models (rows in the order IAM, READ2016, LAM). The line was chosen among those the in-domain model reads perfectly, so its row shows 0% by construction.],
  )
}

== Corrupted test images (RQ4, H3) <sec:robust>

*Setup.* Corruptions are applied to each model's own test set at inference time only; no retraining is involved. H3 predicts that corruptions the training augmentation covers hurt less than those it does not, so the corruptions are grouped accordingly. The training pipeline applies a mild perspective warp, stroke-width changes (erosion/dilation with a kernel of at most 2 px) and colour jitter; it never blurs or adds noise. It does not rotate either, but its perspective warp moves the line corners up and down by up to 8 px, which tilts a 512-px line by up to about 1.8° — so the 2° rotation level sits at the edge of what training shows, and only 5° and 10° are clearly uncovered.

- *Not covered by training:* Gaussian blur (σ = 1, 2, 3 px at model resolution, kernel 2⌈3σ⌉+1), rotation (2°, 5°, 10°; the raw line is rotated with its bounding box expanded before resizing, as a skewed line from a segmenter would look), Gaussian pixel noise (σ = 0.05, 0.1, 0.2).
- *Covered by training:* stroke thickening and thinning with kernels of 2 px (the training range), 3 and 4 px (beyond it).
- *Control:* contrast reduction (factor 0.7, 0.4, 0.2). Section 6 explains why this one cannot test H3.

#let corr-rows = (
  ("clean", "0", "clean", "—"),
  ("blur", "1.0", "blur σ = 1", "no"), ("blur", "2.0", "blur σ = 2", "no"), ("blur", "3.0", "blur σ = 3", "no"),
  ("rotate", "2", "rotate 2°", "no"), ("rotate", "5", "rotate 5°", "no"), ("rotate", "10", "rotate 10°", "no"),
  ("noise", "0.05", "noise σ = 0.05", "no"), ("noise", "0.1", "noise σ = 0.1", "no"), ("noise", "0.2", "noise σ = 0.2", "no"),
  ("thicken", "2", "thicken 2 px", "yes"), ("thicken", "3", "thicken 3 px", "beyond"), ("thicken", "4", "thicken 4 px", "beyond"),
  ("thin", "2", "thin 2 px", "yes"), ("thin", "3", "thin 3 px", "beyond"), ("thin", "4", "thin 4 px", "beyond"),
  ("contrast", "0.7", "contrast 0.7", "control"), ("contrast", "0.4", "contrast 0.4", "control"), ("contrast", "0.2", "contrast 0.2", "control"),
)
#figure(
  table(columns: 5, align: (left, left, right, right, right),
    table.hline(),
    th([Corruption], [In training?], [IAM], [READ2016], [LAM]),
    table.hline(stroke: 0.4pt),
    ..corr-rows.map(((c, s, label, cov)) => (
      label, cov, ..names.map(d => pc(sev(d, c, s)))
    )).flatten(),
    table.hline(),
  ),
  caption: [Test CER under corruption. "beyond" = the corruption type is in the training augmentation, but at a stronger level than training uses. The clean READ2016 value differs from @tab:main by 0.01 points (one or two characters) because GPU floating-point results vary slightly with the batch size.],
) <tab:robust>

#figure(image("figures/robustness.svg", width: 100%), caption: [CER versus corruption severity; severity 0 (or contrast 1.0) is the clean test set.]) <fig:robust>

*Result.* Corruptions covered by training cost almost nothing at the strength training uses: thickening or thinning strokes by 2 px raises CER by only 0.14–0.63 points. Every uncovered corruption costs more even at its mildest level — blur σ = 1: +0.6 to +2.0 points, noise σ = 0.05: +2.4 to +8.3 (and rotation 2°, which training nearly covers: +0.4 to +2.1) — and this holds within each dataset. At moderate levels the uncovered corruptions are destructive: blur σ = 2 gives 18.6–54.8% CER, noise σ = 0.1 gives 25.2–45.7%, and a 10° tilt 20.4–66.0%. *H3 is confirmed.* (Severities of different corruption types have no common scale, so the comparison rests on the mildest level of each type.)

Three further observations qualify the result:

- *Coverage protects only the trained range.* Beyond it, a covered corruption breaks the model as well: thinning strokes by 4 px gives 60.1% CER on IAM and 30.8% on LAM. The augmentation buys robustness to the magnitudes it shows the model, not to the corruption type in general. Thinning is more harmful than thickening, plausibly because thin strokes disappear entirely.
- *Rotation depends on line length.* READ2016 is far more robust to rotation (20.4% at 10°) than IAM (60.7%) and LAM (66.0%). READ2016 lines are short (median 21 characters, against 43 and 39), so tilting them barely enlarges their bounding box, whereas a long tilted line needs a much taller box and its letters shrink when the box is scaled to the fixed 64-pixel input height. The flawed first version, which rotated the padded canvas, reported a uniform 42.8–45.3% for all three datasets — an artifact of clipping that hid this difference.
- *IAM is the most fragile dataset* under almost every corruption, consistent with it having the fewest training lines (6,482).

Contrast reduction leaves CER unchanged to within 0.02 points at every level. This is not robustness learned from data: the input standardization removes any change of the form _a_ · image + _b_ before the network sees it (up to the small constant ε = 10#super[−5] in the normalization), and torchvision's contrast reduction is exactly such a change.

#figure(
  side-by-side(E.corruption, tag: r => [#r.corruption #if r.corruption != "clean" [#r.severity]], h: 0.95cm),
  caption: [One IAM test line as the model actually sees it (64 × 512 input) under each corruption, with the IAM model's transcription. `rotate_canvas` is the flawed first version of the rotation test, which cuts off the line ends (Section 6).],
)

// ---------------------------------------------------------------- 5

= Error analysis: what kind of mistakes?

Every test transcription was aligned with its ground truth by minimum edit distance, and each error was classified as a substitution (wrong character), insertion (extra character) or deletion (missed character).

#figure(
  table(columns: 4, align: (left, right, right, right),
    table.hline(),
    th([], [IAM], [READ2016], [LAM]),
    table.hline(stroke: 0.4pt),
    [Lines without any error], ..names.map(d => pc(A.at(d).exact_lines, d: 1)),
    [Substitutions (share of errors)], ..names.map(d => pc(A.at(d).share.sub, d: 1)),
    [Deletions], ..names.map(d => pc(A.at(d).share.dele, d: 1)),
    [Insertions], ..names.map(d => pc(A.at(d).share.ins, d: 1)),
    table.hline(stroke: 0.4pt),
    table.cell(colspan: 4, text(fill: muted, style: "italic")[Error rate by character class (share of ground-truth characters misread)]),
    ..("lowercase", "uppercase", "accented letter", "digit", "punctuation", "space").map(c => {
      let cells = names.map(d => {
        let hit = A.at(d).by_class.filter(x => x.cls == c)
        if hit.len() == 0 [—] else [#pc(hit.at(0).err, d: 1)]
      })
      ([#c], ..cells)
    }).flatten(),
    table.hline(),
  ),
  caption: [Error types per dataset (classes with fewer than 20 occurrences in a test set are omitted).],
) <tab:errors>

#figure(
  grid(columns: 3, column-gutter: 1.2em,
    ..names.map(d => table(columns: 3, align: (center, center, right),
      table.hline(),
      table.header(table.cell(colspan: 3, strong(full.at(d)))),
      table.hline(stroke: 0.4pt),
      [*GT*], [*read as*], [*n*],
      ..A.at(d).top_subs.map(s => (mono(vis(s.gt)), mono(vis(s.pred)), [#s.n])).flatten(),
      table.hline(),
    ))
  ),
  caption: [Most frequent character substitutions (␣ = space).],
) <tab:subs>

Substitutions make up half to two thirds of all errors, and the confusions are the classic ones of handwriting rather than random noise. On IAM the top pairs are a ↔ o (an open versus closed loop), the minim letters n, m, u and r (letters built from the same short vertical strokes) and t → l. Rare characters are much harder than common ones: uppercase letters (14.5% error) and digits (27.2%) against 4.6% for lowercase, because they occur rarely in the 6,482 training lines. Only 30.9% of IAM test lines are transcribed without any error.

READ2016 shows the same minim and loop confusions (m ↔ n, a ↔ o), plus errors on the abbreviation marks of early modern German (v → ū, ō → o) and on punctuation, which has the highest error rate of any class (17.9%; mostly comma versus full stop). LAM has the largest share of insertions (28.2%), and one of its top "substitutions" is `#` → o. Both are explained by the annotation: LAM replaces struck-out or illegible words with a single `#`, so when the model reads the crossed-out word anyway, every extra letter counts as an insertion. Its worst test line in Section 3.5 ("mi \#" read as "mi ralegao") is exactly this case. Across all datasets spaces are almost always right (1.4–1.7% error), so word segmentation is not the main problem.

// ---------------------------------------------------------------- 6

= Experiment audit: are the results free of bugs?

Before writing this report, every experiment was re-checked against its code, its logs and its data. @tab:audit summarizes the checks. Three problems were found, all in the robustness experiment; they were fixed and that experiment was re-run (the numbers in Section 4.3 are the corrected ones). Nothing that affects the reproduction, the reduced-data or the cross-dataset results was found.

#let ok = text(fill: okc, weight: "bold")[OK]
#let fixed = text(fill: bad, weight: "bold")[Fixed]
#let note = text(fill: muted, weight: "bold")[Note]
#figure(
  table(columns: (auto, 1fr, auto), align: (left, left, center),
    table.hline(),
    th([Check], [Finding], [Status]),
    table.hline(stroke: 0.4pt),
    [Reported numbers], [All test CER/WER values in this report equal the values in the six `test.py` logs, and three independent inference paths (`test.py`, `view_test_results.py`, the report generator) give identical CER. The only discrepancy is 0.01 points on READ2016 between batch sizes, from GPU floating-point rounding.], ok,
    [Model selection], [Checkpoints are selected on validation CER; each test set is evaluated once per model.], ok,
    [Split integrity], [No image file appears in two splits of any dataset. Identical _transcriptions_ do recur between train and test (IAM 0.1%, READ2016 6.9%, LAM 6.1% of test lines), which is inherent to the official splits (short formulaic lines), not introduced here.], ok,
    [Reduced-data subsets], [Each subset is a duplicate-free subset of the official training list, disjoint from validation and test, with exactly the full 89-character alphabet. Subsets were drawn independently rather than nested.], ok,
    [Cross-dataset evaluation], [Checkpoints load with strict key and shape matching (an earlier class-count mismatch was fixed before any results were reported). A full re-run on 14 October reproduced all six pairs to two decimals.], ok,
    [Robustness: contrast], [The model standardizes every input image (`layer_norm` over the whole image, `model/HTR_VT.py`). Contrast reduction, as implemented by torchvision, is an affine change of each image, which this standardization removes up to a negligible numerical constant. The flat contrast curve was earlier credited to the colour-jitter augmentation; that explanation was wrong. Contrast is now reported only as a control.], fixed,
    [Robustness: blur], [The first version used a fixed 5-pixel kernel. For σ = 2 and 3 this truncates the Gaussian to nearly the same box filter, so the labelled severities were overstated and the damage understated (IAM σ = 1/2/3: #V1.iam_blur.values().map(v => fmt(v.v1_cer)).join(" / ")% with the old kernel versus #V1.iam_blur.values().map(v => fmt(v.v2_cer)).join(" / ")% corrected; both reproduced by `report/audit_v1_checks.py`). Kernel is now 2⌈3σ⌉+1.], fixed,
    [Robustness: rotation], [The first version rotated the padded 64 × 512 canvas, which cuts off the line ends: at 10°, #pc(V1.canvas_rotation_ink_lost_pct.at("10"), d: 1) of the ink pixels of the IAM test set left the image (#pc(V1.canvas_rotation_ink_lost_pct.at("5"), d: 1) at 5°), so part of the "error" was deleted text. Now the raw line is rotated with an expanded bounding box. The old variant is kept as `rotate_canvas` to quantify the artifact.], fixed,
    [Robustness: H3 design], [Blur, rotation and contrast alone could not test H3: no corruption in the set was genuinely covered by training. Stroke thickening/thinning (the training's own erosion/dilation) and pixel noise were added.], fixed,
    [Inherited code quirks], [Left unchanged to stay faithful to the authors' code: the training loop runs 99,999 rather than 100,000 iterations; the `--blur` options in `utils/option.py` are never used; the label converter special-cases an 87-character alphabet (READ2016) so that `[` and `]` can never be output; the paper lists elastic distortion among the augmentations, but the released code defines it and never applies it; hue and saturation jitter do nothing on greyscale images, and brightness/contrast jitter are largely cancelled by the input standardization.], note,
    table.hline(),
  ),
  caption: [Audit of the experiments.],
) <tab:audit>

// ---------------------------------------------------------------- 7

== Independent validation by LLM judges

To check the evaluation beyond my own audit, five independent LLM judges (Claude subagents) were each given one part of the evaluation, the raw files, and no access to my conclusions. Each was asked to find discrepancies, not to confirm results.

#figure(
  { set par(justify: false); table(columns: (auto, 0.8fr, 1.5fr), align: (left, left, left),
    table.hline(),
    th([Judge], [Task], [Finding]),
    table.hline(stroke: 0.4pt),
    [Code and metrics], [Read all evaluation code; recompute CER/WER with its own edit-distance implementation; re-run in-domain inference; check every corruption on real images.], [*No critical discrepancy.* All test CER/WER values reproduce (e.g. IAM 6,507 edits / 125,615 characters = 5.180%; all four LAM fractions). All scripts aggregate CER over characters as in the paper, decode identically, and the bootstrap is correctly paired. Four minor issues, all now reflected in this report: training's perspective warp already tilts lines by up to ≈1.8°; the white rotation fill on grey LAM backgrounds; contrast invariance is approximate (ε in the normalization); READ2016's clean CER varies by 0.01 between batch sizes.],
    [Numbers and claims], [Compare every number in this report, `RESULTS.md`, `PROJECT_REPORT.md` and the slide deck with the logs, JSON files and the paper.], [All test, bootstrap, cross-dataset, robustness, epoch and dataset numbers match. Corrected after the check: "LAM is the weakest source" (READ2016 and LAM are equally weak; IAM transfers best), READ2016 training time (≈30 h, not 29.8 h), and loosely stated ranges (now 9–20×, 2.7–4.3×, 18–66%). The slide deck predates the corrected robustness study and is out of date.],
    [Images: IAM, READ2016, LAM], [Look at 74 sampled test images (per dataset 10 random lines, 10 random error lines, the 5 worst) and judge ground truth, prediction and the per-line edit distance.], [*Edit distance correct on all 74 lines.* Labels are reliable on ordinary lines, but several high-error lines are label problems: IAM's worst line is a mis-segmented form title; two IAM and two LAM labels contain errors the model got right (e.g. LAM "mormorado" for the clearly written "mormorando", verified); in READ2016 about half of the remaining errors are uncertain full stops or spaces, and one crop cuts off part of its labelled word.],
    table.hline(),
  ) },
  caption: [Independent validation of the evaluation by LLM judges.],
) <tab:judges>

The image judges also exposed how much LAM's annotation convention costs: struck-out or illegible words are written as a single `#`, and the model transcribes them anyway. The #thou(276) test lines that contain `#` produce 31.7% of all LAM character errors; without them LAM's CER would be 2.26% instead of 3.06%. The reported numbers follow each dataset's standard protocol and remain comparable with the paper, but they overstate the model's actual misreadings somewhat, mostly through a few label and segmentation outliers. LLM judges can themselves misread difficult historical script; their verdicts were used to locate problems, and the two that changed the text of this report were checked by hand.

= Limitations

- *One run per configuration.* Each model was trained once (about one GPU-day each), so the variation between training runs is unknown. The bootstrap intervals cover only test-sample variation, and small differences — notably the 75% vs. 100% LAM gap — should be read with that in mind.
- *Batch size.* All models use batch 32 instead of 128, which plausibly explains most of the small IAM and LAM gaps but was not isolated experimentally.
- *IAM source.* The IAM data comes from the Teklia mirror. Split sizes and alphabet match the official split, but file-level identity could not be verified.
- *Synthetic corruptions.* Blur, rotation, noise and stroke-width changes are approximations of real scanning and writing variation, applied only at test time. Rotation additionally interacts with the fixed 64-pixel input height (a skewed line's bounding box is taller, so the glyphs shrink), and the corners it uncovers are filled with white while LAM and READ2016 have grey parchment backgrounds. An independent re-run of LAM with each image's own background colour as fill gave 15.7% CER at 5° (instead of 13.6%) and 59.6% at 10° (instead of 66.0%): the rotation numbers carry a few points of such artifacts, although no conclusion changes.
- *Scope.* Three Latin-script datasets, pre-segmented lines, greedy CTC decoding without a language model, and no repetition of the paper's component ablations.

= Reproducibility

All code is in the repository. `run_queue.sh` trains and tests the three main models; `run_reduced_data_queue.sh` the reduced-data models (subset lists in `data/LAM/train_{25,50,75}pct.ln`); `cross_eval.py` and `robustness_eval.py` run the two inference-only studies; `view_test_results.py` and `predict.py` show transcriptions for test-set or arbitrary line images; `report/make_figures.py` regenerates every figure, table value and example in this report from the logs and checkpoints, and `report/audit_v1_checks.py` reproduces the numbers quoted for the flawed first robustness version, followed by `typst compile report/report.typ`.

= References

#set par(justify: false)
#set text(9.5pt)
[1] Y. Li, D. Chen, T. Tang, X. Shen. HTR-VT: Handwritten Text Recognition with Vision Transformer. _Pattern Recognition_, 2025. arXiv:2409.08573. \
[2] A. Graves, S. Fernández, F. Gomez, J. Schmidhuber. Connectionist Temporal Classification. _ICML_, 2006. \
[3] P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. Sharpness-Aware Minimization for Efficiently Improving Generalization. _ICLR_, 2021. \
[4] A. Dosovitskiy et al. An Image Is Worth 16x16 Words. _ICLR_, 2021. \
[5] U.-V. Marti, H. Bunke. The IAM-Database. _IJDAR_ 5:39–46, 2002. \
[6] M. Li et al. TrOCR: Transformer-Based Optical Character Recognition with Pre-Trained Models. _AAAI_, 2023.
