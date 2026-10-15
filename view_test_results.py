"""
Generate a browsable HTML report of test-set predictions: each row shows the
line image next to its ground-truth transcription and the model's prediction.

Usage:
    python view_test_results.py --exp iam
    python view_test_results.py --exp lam --n 50 --sort worst
    python view_test_results.py --exp read --n -1          # show every test sample

Runs inference over the full test set (so the printed overall CER/WER match
test.py exactly), then renders --n of the samples to an HTML file you can
open in a browser (images are embedded, so the file is self-contained).
"""
import argparse
import base64
import os
import re
from collections import OrderedDict

import editdistance
import torch

from data import dataset
from model import HTR_VT
from utils import utils

IMG_SIZE = [512, 64]

CHECKPOINTS = {
    'iam': dict(train_ln='./data/iam/train.ln', data_path='./data/iam/lines/',
                test_ln='./data/iam/test.ln', exp='iam', nb_cls=80),
    'read': dict(train_ln='./data/read2016/train.ln', data_path='./data/read2016/lines/',
                 test_ln='./data/read2016/test.ln', exp='read', nb_cls=90),
    'lam': dict(train_ln='./data/LAM/train.ln', data_path='./data/LAM/lines/',
                test_ln='./data/LAM/test.ln', exp='lam', nb_cls=90),
}


def load_model_and_data(name):
    cfg = CHECKPOINTS[name]
    train_dataset = dataset.myLoadDS(cfg['train_ln'], cfg['data_path'], IMG_SIZE)
    test_dataset = dataset.myLoadDS(cfg['test_ln'], cfg['data_path'], IMG_SIZE,
                                     ralph=train_dataset.ralph)

    model = HTR_VT.create_model(nb_cls=cfg['nb_cls'], img_size=IMG_SIZE[::-1])
    ckpt = torch.load(f"./output/{cfg['exp']}/best_CER.pth", map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in ckpt['state_dict_ema'].items():
        model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict, strict=True)
    model = model.cuda().eval()

    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    return model, converter, test_dataset


def run_inference(model, converter, test_dataset, batch_size=16, num_workers=4):
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          pin_memory=True, num_workers=num_workers)
    results = []
    idx = 0
    with torch.no_grad():
        for image_tensors, labels in loader:
            image = image_tensors.cuda()
            preds = model(image).float()
            preds_size = torch.IntTensor([preds.size(1)] * image.size(0))
            preds = preds.permute(1, 0, 2).log_softmax(2)
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

            for pred, gt in zip(preds_str, labels):
                cer = editdistance.eval(pred, gt) / max(len(gt), 1)
                pred_w = utils.format_string_for_wer(pred).split(' ')
                gt_w = utils.format_string_for_wer(gt).split(' ')
                wer = editdistance.eval(pred_w, gt_w) / max(len(gt_w), 1)
                results.append(dict(path=test_dataset.fns[idx], gt=gt, pred=pred, cer=cer, wer=wer))
                idx += 1
    return results


def overall_cer_wer(results):
    tot_ed = sum(editdistance.eval(r['pred'], r['gt']) for r in results)
    tot_len = sum(len(r['gt']) for r in results)
    tot_ed_w, tot_len_w = 0, 0
    for r in results:
        pred_w = utils.format_string_for_wer(r['pred']).split(' ')
        gt_w = utils.format_string_for_wer(r['gt']).split(' ')
        tot_ed_w += editdistance.eval(pred_w, gt_w)
        tot_len_w += len(gt_w)
    return tot_ed / tot_len, tot_ed_w / tot_len_w


def img_to_data_uri(path):
    ext = os.path.splitext(path)[1].lower()
    mime = 'image/png' if ext == '.png' else 'image/jpeg'
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('ascii')
    return f'data:{mime};base64,{data}'


def build_html(exp_name, overall_cer, overall_wer, n_total, rows):
    row_html = []
    for r in rows:
        correct = r['cer'] == 0
        row_class = 'ok' if correct else ('bad' if r['cer'] > 0.3 else 'warn')
        row_html.append(f'''
        <tr class="{row_class}">
          <td><img src="{img_to_data_uri(r['path'])}" class="line-img"></td>
          <td class="mono">{r['gt']}</td>
          <td class="mono">{r['pred']}</td>
          <td>{r['cer']*100:.1f}%</td>
          <td>{r['wer']*100:.1f}%</td>
        </tr>''')

    return f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>HTR-VT test results — {exp_name}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 20px; }}
  .summary {{ margin-bottom: 16px; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: middle; font-size: 13px; }}
  th {{ background: #1E2761; color: #fff; position: sticky; top: 0; }}
  .mono {{ font-family: "Courier New", monospace; white-space: pre-wrap; }}
  .line-img {{ max-width: 420px; max-height: 60px; image-rendering: pixelated; }}
  tr.ok td {{ background: #eafaf0; }}
  tr.warn td {{ background: #fff9e6; }}
  tr.bad td {{ background: #fdeaea; }}
</style>
</head><body>
<h1>HTR-VT test results — {exp_name.upper()} checkpoint</h1>
<div class="summary">
  Overall test set ({n_total} lines): <b>CER {overall_cer*100:.2f}%</b> &nbsp;|&nbsp; <b>WER {overall_wer*100:.2f}%</b><br>
  Showing {len(rows)} sample(s) below. Green = exact match, yellow = minor errors, red = CER &gt; 30%.
</div>
<table>
<tr><th>Image</th><th>Ground truth</th><th>Prediction</th><th>CER</th><th>WER</th></tr>
{''.join(row_html)}
</table>
</body></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['iam', 'lam', 'read'], default='iam')
    parser.add_argument('--n', type=int, default=30,
                         help='number of samples to render in the HTML (-1 = all)')
    parser.add_argument('--sort', choices=['order', 'best', 'worst'], default='order',
                         help='order = test-set order, best/worst = sorted by CER')
    parser.add_argument('--out', default=None, help='output HTML path')
    args = parser.parse_args()

    model, converter, test_dataset = load_model_and_data(args.exp)
    print(f'Running inference on {len(test_dataset)} test lines...')
    results = run_inference(model, converter, test_dataset)

    cer, wer = overall_cer_wer(results)
    print(f'Overall: CER {cer*100:.2f}%  WER {wer*100:.2f}%')

    if args.sort == 'best':
        results.sort(key=lambda r: r['cer'])
    elif args.sort == 'worst':
        results.sort(key=lambda r: -r['cer'])

    rows = results if args.n < 0 else results[:args.n]
    out_path = args.out or f'test_results_{args.exp}.html'
    html = build_html(args.exp, cer, wer, len(results), rows)
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'Wrote {out_path} ({len(rows)} samples). Open it in a browser to view.')
