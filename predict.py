"""
Run the trained HTR-VT model on your own handwriting line images.

Usage:
    python predict.py path/to/image1.png path/to/image2.jpg ...
    python predict.py path/to/a/folder/           # predicts every image in the folder

Each image should be a single line of handwriting (not a full page). The
image can be a photo or scan, color or grayscale, any resolution — it gets
resized/padded to the model's expected 512x64 input automatically.

Defaults to the IAM checkpoint (modern English handwriting). Pass --exp lam
or --exp read to use the LAM or READ2016 checkpoints instead.
"""
import argparse
import glob
import os
import re
from collections import OrderedDict

import numpy as np
import skimage
import torch
from PIL import Image

from data import dataset
from model import HTR_VT
from utils import utils

IMG_SIZE = [512, 64]

CHECKPOINTS = {
    'iam': dict(train_ln='./data/iam/train.ln', data_path='./data/iam/lines/',
                exp='iam', nb_cls=80),
    'read': dict(train_ln='./data/read2016/train.ln', data_path='./data/read2016/lines/',
                 exp='read', nb_cls=90),
    'lam': dict(train_ln='./data/LAM/train.ln', data_path='./data/LAM/lines/',
                exp='lam', nb_cls=90),
}

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_model_and_converter(name):
    cfg = CHECKPOINTS[name]
    train_dataset = dataset.myLoadDS(cfg['train_ln'], cfg['data_path'], IMG_SIZE)
    model = HTR_VT.create_model(nb_cls=cfg['nb_cls'], img_size=IMG_SIZE[::-1])

    ckpt_path = f"./output/{cfg['exp']}/best_CER.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in ckpt['state_dict_ema'].items():
        model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict, strict=True)
    model = model.cuda().eval()

    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    return model, converter


def load_image(path, max_w, max_h):
    image_data = np.array(Image.open(path).convert('L'))

    x, y = image_data.shape[:2]
    y = min(int(y * max_h / x), max_w)
    x = max_h
    image_data = np.array(Image.fromarray(image_data).resize((y, x)))

    image_data = skimage.img_as_float32(image_data)
    image_data = np.expand_dims(image_data, axis=-1)
    image_data = np.pad(image_data, ((0, 0), (0, max_w - image_data.shape[1]), (0, 0)),
                         mode='constant', constant_values=(1.0))

    tensor = torch.from_numpy(image_data.transpose((2, 0, 1))).unsqueeze(0).float()
    return tensor


def predict(model, converter, image_paths):
    results = []
    with torch.no_grad():
        for path in image_paths:
            image = load_image(path, IMG_SIZE[0], IMG_SIZE[1]).cuda()
            preds = model(image).float()
            preds_size = torch.IntTensor([preds.size(1)])
            preds = preds.permute(1, 0, 2).log_softmax(2)
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            text = converter.decode(preds_index.data, preds_size.data)[0]
            results.append((path, text))
    return results


def collect_image_paths(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in IMAGE_EXTS:
                paths.extend(sorted(glob.glob(os.path.join(item, f'*{ext}'))))
        else:
            paths.append(item)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='image file(s) or a folder of images')
    parser.add_argument('--exp', choices=['iam', 'lam', 'read'], default='iam',
                         help='which trained checkpoint to use (default: iam)')
    args = parser.parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        raise SystemExit('No images found.')

    model, converter = load_model_and_converter(args.exp)
    for path, text in predict(model, converter, image_paths):
        print(f'{path}:\n  {text}\n')
