"""
Test-time corruption robustness of the trained HTR-VT checkpoints (RQ4 / H3).

Corruptions are grouped by whether the training pipeline covers them:
  not covered by training augmentation: blur, rotate, noise
  covered (training applies stroke-width changes with kernel <= 2): thicken, thin
  control: contrast (cancelled exactly by the model's input standardization)
  rotate_canvas: the v1 rotation, applied to the padded 512x64 canvas, which
                 clips the line ends; kept only to quantify that artifact.

Writes robustness_results.json and prints a table.
"""
import json
import math

import numpy as np
import torch
import torchvision.transforms.functional as TF
import editdistance
from PIL import Image
from collections import OrderedDict

from utils import utils
from data import dataset
from data.transform import Erosion, Dilation
from model import HTR_VT

IMG_SIZE = [512, 64]

DATASETS = {
    'IAM': dict(train_ln='./data/iam/train.ln', data_path='./data/iam/lines/',
                test_ln='./data/iam/test.ln', exp='iam', nb_cls=80),
    'READ': dict(train_ln='./data/read2016/train.ln', data_path='./data/read2016/lines/',
                 test_ln='./data/read2016/test.ln', exp='read', nb_cls=90),
    'LAM': dict(train_ln='./data/LAM/train.ln', data_path='./data/LAM/lines/',
                test_ln='./data/LAM/test.ln', exp='lam', nb_cls=90),
}

# stage: 'raw' = PIL image before resizing, 'model' = PIL image at 512x64, 'tensor' = batch tensor
CORRUPTIONS = {
    'clean': ('tensor', lambda x, s: x),
    'blur': ('tensor', lambda x, s: TF.gaussian_blur(x, kernel_size=2 * math.ceil(3 * s) + 1, sigma=s)),
    'rotate': ('raw', lambda im, s: im.rotate(s, resample=Image.BILINEAR, expand=True, fillcolor=255)),
    'noise': ('tensor', lambda x, s: (x + s * torch.randn_like(x)).clamp(0, 1)),
    'thicken': ('model', lambda im, s: Erosion((s, s), 1)(im)),
    'thin': ('model', lambda im, s: Dilation((s, s), 1)(im)),
    'contrast': ('tensor', lambda x, s: TF.adjust_contrast(x, contrast_factor=s)),
    'rotate_canvas': ('tensor', lambda x, s: TF.rotate(x, angle=s, fill=1.0)),
}

SEVERITIES = {
    'clean': [0],
    'blur': [1.0, 2.0, 3.0],
    'rotate': [2, 5, 10],
    'noise': [0.05, 0.1, 0.2],
    'thicken': [2, 3, 4],
    'thin': [2, 3, 4],
    'contrast': [0.7, 0.4, 0.2],
    'rotate_canvas': [2, 5, 10],
}


def preprocess(im, max_w=IMG_SIZE[0], max_h=IMG_SIZE[1]):
    """Same resize-to-height + white right-padding as data/dataset.py:get_images."""
    arr = np.array(im)
    h, w = arr.shape[:2]
    new_w = min(int(w * max_h / h), max_w)
    arr = np.array(Image.fromarray(arr).resize((new_w, max_h)))
    arr = arr.astype(np.float32) / 255.
    return np.pad(arr, ((0, 0), (0, max_w - new_w)), mode='constant', constant_values=1.0)


class CorruptedLines(torch.utils.data.Dataset):
    def __init__(self, fns, labels, stage, fn, severity):
        self.fns, self.labels = fns, labels
        self.stage, self.fn, self.severity = stage, fn, severity

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, i):
        im = Image.open(self.fns[i]).convert('L')
        if self.stage == 'raw':
            im = self.fn(im, self.severity)
        arr = preprocess(im)
        if self.stage == 'model':
            arr = np.array(self.fn(Image.fromarray(np.uint8(arr * 255)), self.severity)).astype(np.float32) / 255.
        return torch.from_numpy(arr).unsqueeze(0), self.labels[i]


def load_model_and_converter(name):
    cfg = DATASETS[name]
    train_dataset = dataset.myLoadDS(cfg['train_ln'], cfg['data_path'], IMG_SIZE)
    model = HTR_VT.create_model(nb_cls=cfg['nb_cls'], img_size=IMG_SIZE[::-1])
    ckpt = torch.load(f"./output/{cfg['exp']}/best_CER.pth", map_location='cpu')
    model_dict = OrderedDict()
    for k, v in ckpt['state_dict_ema'].items():
        model_dict[k.replace('module.', '')] = v
    model.load_state_dict(model_dict, strict=True)
    model = model.cuda().eval()
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    return model, converter


def decode(model, converter, image):
    preds = model(image.cuda()).float()
    preds_size = torch.IntTensor([preds.size(1)] * image.size(0))
    preds = preds.permute(1, 0, 2).log_softmax(2)
    _, preds_index = preds.max(2)
    preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
    return converter.decode(preds_index.data, preds_size.data)


def evaluate(name, model, converter, corruption, severity):
    cfg = DATASETS[name]
    test_dataset = dataset.myLoadDS(cfg['test_ln'], cfg['data_path'], IMG_SIZE)
    stage, fn = CORRUPTIONS[corruption]
    ds = CorruptedLines(test_dataset.fns, test_dataset.tlbls, stage, fn, severity)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, pin_memory=True, num_workers=8)

    torch.manual_seed(0)
    tot_ED, length_of_gt, tot_ED_wer, length_of_gt_wer = 0, 0, 0, 0
    with torch.no_grad():
        for image_tensors, labels in loader:
            if stage == 'tensor':
                image_tensors = fn(image_tensors, severity)
            preds_str = decode(model, converter, image_tensors)
            for pred, gt in zip(preds_str, labels):
                tot_ED += editdistance.eval(pred, gt)
                length_of_gt += len(gt)
                pred_w = utils.format_string_for_wer(pred).split(' ')
                gt_w = utils.format_string_for_wer(gt).split(' ')
                tot_ED_wer += editdistance.eval(pred_w, gt_w)
                length_of_gt_wer += len(gt_w)
    return tot_ED / length_of_gt, tot_ED_wer / length_of_gt_wer


if __name__ == '__main__':
    results = []
    print(f"{'dataset':<6} {'corruption':<14} {'severity':>9} {'CER':>8} {'WER':>8}", flush=True)
    for name in DATASETS:
        model, converter = load_model_and_converter(name)
        for corruption, sevs in SEVERITIES.items():
            for sev in sevs:
                cer, wer = evaluate(name, model, converter, corruption, sev)
                results.append(dict(dataset=name, corruption=corruption, severity=sev, cer=cer, wer=wer))
                print(f"{name:<6} {corruption:<14} {sev:>9} {cer*100:7.2f}% {wer*100:7.2f}%", flush=True)
                with open('robustness_results.json', 'w') as f:
                    json.dump(results, f, indent=1)
