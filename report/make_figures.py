"""
Build every figure and data file used by report/report.typ.

Run from the repo root:
    python report/make_figures.py                 # inference examples + error analysis + plots
    python report/make_figures.py --plots-only    # re-draw plots from existing logs/JSON

Outputs go to report/figures/: *.svg plots, line-image PNGs, metrics.json, examples.json.
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.getcwd())

FIG = 'report/figures'
NAVY, STEEL, ICE, GREY = '#1E2761', '#6E7A99', '#9DB4E6', '#B8B8C2'
DS_COLORS = {'IAM': '#1E2761', 'READ': '#C0504D', 'LAM': '#4F8A5B'}

PAPER = {'IAM': (4.7, 14.9), 'READ': (3.9, 16.5), 'LAM': (2.8, 7.4)}
EXP = {'IAM': 'iam', 'READ': 'read', 'LAM': 'lam'}
LISTS = {
    'IAM': ('./data/iam/train.ln', './data/iam/lines/', './data/iam/test.ln', 80),
    'READ': ('./data/read2016/train.ln', './data/read2016/lines/', './data/read2016/test.ln', 90),
    'LAM': ('./data/LAM/train.ln', './data/LAM/lines/', './data/LAM/test.ln', 90),
}
FRACTIONS = [(25, 'lam_25pct'), (50, 'lam_50pct'), (75, 'lam_75pct'), (100, 'lam')]
SEVERITY_KEYS = ['clean', 'blur', 'rotate', 'noise', 'thicken', 'thin', 'contrast', 'rotate_canvas']

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#E3E3E8', 'grid.linewidth': 0.6,
                     'svg.fonttype': 'path'})


# ---------------------------------------------------------------- log parsing

def test_metrics(exp):
    line = [l for l in open(f'output_{exp}_test.log') if 'Test.' in l][-1]
    cer, wer = re.findall(r'CER : ([\d.]+).*WER : ([\d.]+)', line)[0]
    return float(cer) * 100, float(wer) * 100


def val_curve(exp):
    cers = [float(re.findall(r'CER : ([\d.]+)', l)[0]) * 100
            for l in open(f'output/{exp}/run.log') if 'Val. loss' in l]
    return [1000 * (i + 1) for i in range(len(cers))], cers


def best_val(exp):
    return min(val_curve(exp)[1])


def cross_results():
    path = 'output_cross_eval_check.log'
    rows = {}
    for l in open(path):
        m = re.match(r'(\w+)\s+(\w+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%', l)
        if m:
            rows[(m[1], m[2])] = dict(cer=float(m[3]), wer=float(m[4]), oov=float(m[5]))
    return rows


# ---------------------------------------------------------------- plots

def plot_curves(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ds in ['IAM', 'READ', 'LAM']:
        it, c = val_curve(EXP[ds])
        axes[0].plot(np.array(it) / 1000, c, color=DS_COLORS[ds], lw=1.3, label=ds)
    axes[0].set(ylim=(0, 15), xlabel='iteration (thousands)', ylabel='validation CER (%)',
                title='Main runs (bs = 32, 100k iterations)')
    axes[0].legend(frameon=False)
    shades = ['#B9C3E0', '#8796C4', '#55679F', NAVY]
    for (pct, exp), col in zip(FRACTIONS, shades):
        it, c = val_curve(exp)
        axes[1].plot(np.array(it) / 1000, c, color=col, lw=1.3, label=f'{pct}% of LAM')
    axes[1].set(ylim=(2, 8), xlabel='iteration (thousands)', title='LAM reduced-data runs')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'{FIG}/val_curves.svg', bbox_inches='tight')
    plt.close(fig)


def plot_reproduction(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    x = np.arange(3)
    names = ['IAM', 'READ', 'LAM']
    for ax, k, lab in [(axes[0], 0, 'CER (%)'), (axes[1], 1, 'WER (%)')]:
        paper = [PAPER[n][k] for n in names]
        ours = [metrics['main'][n]['cer' if k == 0 else 'wer'] for n in names]
        b1 = ax.bar(x - 0.18, paper, 0.34, color=GREY, label='paper (bs = 128)')
        b2 = ax.bar(x + 0.18, ours, 0.34, color=NAVY, label='ours (bs = 32)')
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f'{b.get_height():.2f}',
                        ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(x, ['IAM', 'READ2016', 'LAM'])
        ax.set_ylabel(lab)
        ax.grid(axis='x', visible=False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/reproduction.svg', bbox_inches='tight')
    plt.close(fig)


def plot_reduced(metrics):
    pct = [r['pct'] for r in metrics['reduced']]
    cer = [r['cer'] for r in metrics['reduced']]
    wer = [r['wer'] for r in metrics['reduced']]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(pct, cer, 'o-', color=NAVY, lw=1.6, label='CER')
    for p, c in zip(pct, cer):
        ax.annotate(f'{c:.2f}', (p, c), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(pct, wer, 's--', color=STEEL, lw=1.2, label='WER')
    ax2.spines['right'].set_visible(True)
    ax2.grid(False)
    ax.set(xlabel='share of LAM training lines (%)', ylabel='test CER (%)', xticks=pct, ylim=(2.8, 4.1))
    ax2.set(ylabel='test WER (%)', ylim=(7.5, 11.5))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{FIG}/reduced_data.svg', bbox_inches='tight')
    plt.close(fig)


def plot_cross(metrics):
    names = ['IAM', 'READ', 'LAM']
    m = np.array([[metrics['cross'][s][t]['cer'] for t in names] for s in names])
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.grid(False)
    im = ax.imshow(m, cmap='Blues', vmin=0, vmax=85)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{m[i, j]:.1f}', ha='center', va='center',
                    color='white' if m[i, j] > 45 else 'black', fontsize=10,
                    fontweight='bold' if i == j else 'normal')
    labels = ['IAM', 'READ2016', 'LAM']
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel('test set (target)')
    ax.set_ylabel('trained on (source)')
    fig.colorbar(im, ax=ax, fraction=0.046, label='CER (%)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/cross_heatmap.svg', bbox_inches='tight')
    plt.close(fig)


def plot_robustness(metrics):
    rob = metrics.get('robustness')
    complete = rob and all(c in rob.get(ds, {}) for ds in ['IAM', 'READ', 'LAM'] for c in SEVERITY_KEYS)
    if not complete:
        print('robustness_results.json not found, skipping robustness plot')
        return
    panels = [('blur', 'Gaussian blur σ', False), ('rotate', 'rotation ° (line bounding box)', False),
              ('noise', 'Gaussian noise σ', False), ('thicken', 'stroke thickening (kernel px)', True),
              ('thin', 'stroke thinning (kernel px)', True), ('contrast', 'contrast factor', None)]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharey=False)
    for ax, (corr, xlabel, covered) in zip(axes.flat, panels):
        for ds in ['IAM', 'READ', 'LAM']:
            clean = rob[ds]['clean']['0']['cer']
            sev = list(rob[ds][corr].keys())
            xs = [0] + [float(s) for s in sev] if corr != 'contrast' else [1.0] + [float(s) for s in sev]
            ys = [clean] + [rob[ds][corr][s]['cer'] for s in sev]
            ax.plot(xs, ys, 'o-', color=DS_COLORS[ds], lw=1.3, ms=3.5, label=ds)
        if corr == 'contrast':
            ax.invert_xaxis()
        tag = {True: 'covered by training aug.', False: 'not covered by training aug.',
               None: 'cancelled by input normalization'}[covered]
        ax.set_title(f'{corr}\n({tag})', fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('CER (%)', fontsize=8)
        if corr == 'contrast':
            ax.set_ylim(0, 10)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/robustness.svg', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- inference helpers

def load(exp, train_ln, data_path, nb_cls):
    import torch
    from collections import OrderedDict
    from data import dataset
    from model import HTR_VT
    from utils import utils
    train_dataset = dataset.myLoadDS(train_ln, data_path, [512, 64])
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=[64, 512])
    ckpt = torch.load(f'./output/{exp}/best_CER.pth', map_location='cpu')
    sd = OrderedDict((k.replace('module.', ''), v) for k, v in ckpt['state_dict_ema'].items())
    model.load_state_dict(sd, strict=True)
    return model.cuda().eval(), utils.CTCLabelConverter(train_dataset.ralph.values())


def predict_arrays(model, converter, arrays, bs=32):
    import torch
    from robustness_eval import decode
    out = []
    with torch.no_grad():
        for i in range(0, len(arrays), bs):
            batch = torch.stack([torch.from_numpy(a).unsqueeze(0) for a in arrays[i:i + bs]]).float()
            out.extend(decode(model, converter, batch))
    return out


def load_arrays(fns):
    from robustness_eval import preprocess
    return [preprocess(Image.open(f).convert('L')) for f in fns]


def test_set(ds):
    from data import dataset
    _, dp, test_ln, _ = LISTS[ds]
    t = dataset.myLoadDS(test_ln, dp, [512, 64])
    return t.fns, t.tlbls


def align(gt, pred):
    """Levenshtein alignment -> list of (op, gt_char, pred_char), op in M/S/I/D."""
    n, m = len(gt), len(pred)
    d = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        gi, prev, cur = gt[i - 1], d[i - 1], d[i]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (gi != pred[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (gt[i - 1] != pred[j - 1]):
            ops.append(('M' if gt[i - 1] == pred[j - 1] else 'S', gt[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(('D', gt[i - 1], ''))
            i -= 1
        else:
            ops.append(('I', '', pred[j - 1]))
            j -= 1
    return ops[::-1]


def runs(chars):
    """[(char, bad)] -> merged [[text, bad]] runs for compact JSON."""
    out = []
    for c, bad in chars:
        if out and out[-1][1] == bad:
            out[-1][0] += c
        else:
            out.append([c, bad])
    return out


def annotate(gt, pred):
    import editdistance
    ops = align(gt, pred)
    gt_runs = runs([(g, op != 'M') for op, g, p in ops if op != 'I'])
    pred_runs = runs([(p, op != 'M') for op, g, p in ops if op != 'D'])
    return dict(gt=gt, pred=pred, gt_runs=gt_runs, pred_runs=pred_runs,
                cer=100 * editdistance.eval(pred, gt) / max(len(gt), 1))


def save_line(src, name, height=72):
    im = src if isinstance(src, Image.Image) else Image.open(src).convert('L')
    if im.height > height:
        im = im.resize((max(1, int(im.width * height / im.height)), height), Image.LANCZOS)
    im.save(f'{FIG}/{name}.png', optimize=True)
    return f'figures/{name}.png'


def char_class(c):
    if c == ' ':
        return 'space'
    if c.isdigit():
        return 'digit'
    if c.isalpha():
        if unicodedata.decomposition(c):
            return 'accented letter'
        return 'uppercase' if c.isupper() else 'lowercase'
    return 'punctuation'


def error_analysis(gts, preds):
    ops_total = Counter()
    subs = Counter()
    cls_n, cls_err = Counter(), Counter()
    exact = 0
    for gt, pred in zip(gts, preds):
        exact += gt == pred
        for op, g, p in align(gt, pred):
            ops_total[op] += 1
            if op == 'S':
                subs[(g, p)] += 1
            if op != 'I':
                cls_n[char_class(g)] += 1
                cls_err[char_class(g)] += op != 'M'
    n_err = ops_total['S'] + ops_total['I'] + ops_total['D']
    return dict(
        exact_lines=100 * exact / len(gts),
        n_lines=len(gts),
        share=dict(sub=100 * ops_total['S'] / n_err, ins=100 * ops_total['I'] / n_err,
                   dele=100 * ops_total['D'] / n_err),
        top_subs=[dict(gt=g, pred=p, n=n) for (g, p), n in subs.most_common(8)],
        by_class=[dict(cls=c, n=cls_n[c], err=100 * cls_err[c] / cls_n[c])
                  for c in ['lowercase', 'uppercase', 'accented letter', 'digit', 'punctuation', 'space']
                  if cls_n[c] >= 20],
    )


def pick_line(cands, gts, lo=None, hi=None):
    """Deterministically pick a line from candidates, by default between the dataset's
    50th and 90th length percentile so examples are representative but not tiny."""
    lens = [len(g) for g in gts]
    lo = np.percentile(lens, 50) if lo is None else lo
    hi = np.percentile(lens, 90) if hi is None else hi
    mid = [i for i in cands if lo <= len(gts[i]) <= hi] or list(cands)
    return mid[len(mid) // 2]


# ---------------------------------------------------------------- inference sections

def run_inference_sections():
    import torch
    import robustness_eval as R
    import editdistance

    examples = {'qualitative': {}, 'cross': {}, 'corruption': [], 'reduced': None}
    analysis = {}
    models = {ds: load(EXP[ds], *LISTS[ds][:2], LISTS[ds][3]) for ds in LISTS}
    data = {}
    for ds in LISTS:
        fns, gts = test_set(ds)
        arrays = load_arrays(fns)
        preds = predict_arrays(*models[ds], arrays)
        data[ds] = (fns, gts, arrays, preds)
        cers = np.array([editdistance.eval(p, g) / max(len(g), 1) for p, g in zip(preds, gts)])
        print(ds, 'in-domain CER check: %.2f%%' %
              (100 * sum(editdistance.eval(p, g) for p, g in zip(preds, gts)) / sum(len(g) for g in gts)))
        analysis[ds] = error_analysis(gts, preds)
        json.dump(dict(fns=fns, gts=gts, preds=preds), open(f'{FIG}/predictions_{ds}.json', 'w'),
                  ensure_ascii=False)

        order = np.argsort(cers, kind='stable')
        nonzero = [i for i in order if cers[i] > 0]
        picks = [
            ('Exact match', pick_line([i for i in order if cers[i] == 0], gts)),
            ('Typical error', nonzero[len(nonzero) // 2]),
            ('Hard case (90th percentile)', nonzero[int(len(nonzero) * 0.9)]),
            ('Worst line in test set', int(order[-1])),
        ]
        rows = []
        for k, (label, i) in enumerate(picks):
            row = annotate(gts[i], preds[i])
            row.update(label=label, img=save_line(fns[i], f'qual_{ds}_{k}'), file=os.path.basename(fns[i]))
            rows.append(row)
        examples['qualitative'][ds] = rows

    # cross-dataset: one line per target, transcribed by all three models
    for tgt in LISTS:
        fns, gts, arrays, preds = data[tgt]
        ok = [i for i in range(len(gts)) if preds[i] == gts[i]]
        i = pick_line(ok, gts)
        rows = []
        for src in LISTS:
            p = predict_arrays(*models[src], [arrays[i]])[0]
            r = annotate(gts[i], p)
            r['source'] = src
            rows.append(r)
        examples['cross'][tgt] = dict(img=save_line(fns[i], f'cross_{tgt}'), gt=gts[i], rows=rows)

    # corruptions: one clean-correct IAM line, model-input image shown for each corruption
    fns, gts, arrays, preds = data['IAM']
    ok = [i for i in range(len(gts)) if preds[i] == gts[i]]
    i = pick_line(ok, gts, 30, 55)
    plan = [('clean', 0), ('blur', 2.0), ('blur', 3.0), ('noise', 0.1), ('noise', 0.2),
            ('rotate', 5), ('rotate', 10), ('rotate_canvas', 10), ('thicken', 3), ('thin', 3),
            ('contrast', 0.2)]
    for corr, sev in plan:
        stage, fn = R.CORRUPTIONS[corr]
        torch.manual_seed(0)
        ds = R.CorruptedLines([fns[i]], [gts[i]], stage, fn, sev)
        x, _ = ds[0]
        x = x.unsqueeze(0)
        if stage == 'tensor':
            x = fn(x, sev)
        with torch.no_grad():
            p = R.decode(models['IAM'][0], models['IAM'][1], x)[0]
        arr = (x[0, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
        r = annotate(gts[i], p)
        r.update(corruption=corr, severity=sev,
                 img=save_line(Image.fromarray(arr), f'corr_{corr}_{str(sev).replace(".", "_")}', height=64))
        examples['corruption'].append(r)

    # reduced data: same LAM lines decoded by the four LAM models
    del models
    torch.cuda.empty_cache()
    fns, gts, arrays, _ = data['LAM']
    frac_preds = {}
    for pct, exp in FRACTIONS:
        train_ln = './data/LAM/train.ln' if pct == 100 else f'./data/LAM/train_{pct}pct.ln'
        m, c = load(exp, train_ln, './data/LAM/lines/', 90)
        frac_preds[pct] = predict_arrays(m, c, arrays)
        del m
        torch.cuda.empty_cache()
    # lines whose error count drops at every step and that the 100% model reads perfectly;
    # '#' marks struck-out text in LAM's annotation, so those lines are excluded
    ed = {p: [editdistance.eval(frac_preds[p][i], gts[i]) for i in range(len(gts))] for p, _ in FRACTIONS}
    pcts = [p for p, _ in FRACTIONS]
    strict = [i for i in range(len(gts))
              if all(ed[a][i] > ed[b][i] for a, b in zip(pcts, pcts[1:]))]
    chosen = sorted([i for i in strict if ed[100][i] == 0 and '#' not in gts[i]],
                    key=lambda i: -ed[25][i])[:3]
    examples['reduced'] = dict(lines=[
        dict(img=save_line(fns[i], f'reduced_LAM_{k}'), gt=gts[i], errors=[ed[p][i] for p in pcts],
             rows=[dict(pct=p, **annotate(gts[i], frac_preds[p][i])) for p in pcts])
        for k, i in enumerate(chosen)])
    better = sum(ed[100][i] < ed[25][i] for i in range(len(gts)))
    worse = sum(ed[100][i] > ed[25][i] for i in range(len(gts)))
    examples['reduced'].update(
        exact_lines={p: 100 * np.mean([a == b for a, b in zip(frac_preds[p], gts)]) for p in pcts},
        n_lines=len(gts), n_strict=len(strict), n_better=better, n_worse=worse,
        n_same=len(gts) - better - worse)

    # paired bootstrap over test lines (95% intervals, 2000 resamples)
    def edits(preds, gts):
        return (np.array([editdistance.eval(p, g) for p, g in zip(preds, gts)], dtype=np.float64),
                np.array([len(g) for g in gts], dtype=np.float64))

    rng = np.random.default_rng(0)
    boot = {'main': {}, 'reduced': {}, 'reduced_diff': []}
    for ds in LISTS:
        _, g, _, p = data[ds]
        e, n = edits(p, g)
        idx = rng.integers(0, len(e), (2000, len(e)))
        cer = 100 * e[idx].sum(1) / n[idx].sum(1)
        boot['main'][ds] = [float(np.percentile(cer, 2.5)), float(np.percentile(cer, 97.5))]
    e_frac = {pct: edits(frac_preds[pct], gts) for pct, _ in FRACTIONS}
    idx = rng.integers(0, len(gts), (2000, len(gts)))
    samples = {pct: 100 * e[idx].sum(1) / n[idx].sum(1) for pct, (e, n) in e_frac.items()}
    for pct in samples:
        boot['reduced'][pct] = [float(np.percentile(samples[pct], 2.5)), float(np.percentile(samples[pct], 97.5))]
    for a, b in [(25, 50), (50, 75), (75, 100)]:
        d = samples[a] - samples[b]
        boot['reduced_diff'].append(dict(a=a, b=b, mean=float(d.mean()), lo=float(np.percentile(d, 2.5)),
                                         hi=float(np.percentile(d, 97.5)), p_le0=float((d <= 0).mean())))
    json.dump(boot, open(f'{FIG}/bootstrap.json', 'w'), indent=1)

    json.dump(examples, open(f'{FIG}/examples.json', 'w'), ensure_ascii=False, indent=1)
    json.dump(analysis, open(f'{FIG}/error_analysis.json', 'w'), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- main

def collect_metrics():
    def lines(p):
        return sum(1 for l in open(p) if l.strip())

    metrics = {'main': {}, 'reduced': [], 'cross': {}, 'robustness': {}}
    for ds in ['IAM', 'READ', 'LAM']:
        cer, wer = test_metrics(EXP[ds])
        metrics['main'][ds] = dict(cer=cer, wer=wer, paper_cer=PAPER[ds][0], paper_wer=PAPER[ds][1],
                                   best_val_cer=best_val(EXP[ds]), n_train=lines(LISTS[ds][0]),
                                   epochs=100000 * 32 / lines(LISTS[ds][0]))
    for pct, exp in FRACTIONS:
        cer, wer = test_metrics(exp)
        ln = './data/LAM/train.ln' if pct == 100 else f'./data/LAM/train_{pct}pct.ln'
        n = lines(ln)
        metrics['reduced'].append(dict(pct=pct, n=n, epochs=100000 * 32 / n, cer=cer, wer=wer,
                                       best_val_cer=best_val(exp)))
    cross = cross_results()
    for s in ['IAM', 'READ', 'LAM']:
        metrics['cross'][s] = {}
        for t in ['IAM', 'READ', 'LAM']:
            if s == t:
                metrics['cross'][s][t] = dict(cer=metrics['main'][s]['cer'], wer=metrics['main'][s]['wer'], oov=0.0)
            else:
                metrics['cross'][s][t] = cross[(s, t)]
    if os.path.exists('robustness_results.json'):
        for r in json.load(open('robustness_results.json')):
            d = metrics['robustness'].setdefault(r['dataset'], {}).setdefault(r['corruption'], {})
            d[str(r['severity'])] = dict(cer=100 * r['cer'], wer=100 * r['wer'])
    json.dump(metrics, open(f'{FIG}/metrics.json', 'w'), indent=1)
    return metrics


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plots-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    metrics = collect_metrics()
    plot_curves(metrics)
    plot_reproduction(metrics)
    plot_reduced(metrics)
    plot_cross(metrics)
    plot_robustness(metrics)
    if not args.plots_only:
        run_inference_sections()
    print('done')
