import torch
import torchvision.transforms.functional as TF
import editdistance
from collections import OrderedDict

from utils import utils
from data import dataset
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

CORRUPTIONS = {
    'clean': lambda x, sev: x,
    'blur': lambda x, sev: TF.gaussian_blur(x, kernel_size=5, sigma=sev),
    'rotate': lambda x, sev: TF.rotate(x, angle=sev, fill=1.0),
    'contrast': lambda x, sev: TF.adjust_contrast(x, contrast_factor=sev),
}

SEVERITIES = {
    'clean': [0],
    'blur': [1.0, 2.0, 3.0],
    'rotate': [2, 5, 10],
    'contrast': [0.7, 0.4, 0.2],
}


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


def evaluate(name, model, converter, corruption, severity):
    cfg = DATASETS[name]
    test_dataset = dataset.myLoadDS(cfg['test_ln'], cfg['data_path'], IMG_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,
                                               pin_memory=True, num_workers=4)

    tot_ED, length_of_gt = 0, 0
    tot_ED_wer, length_of_gt_wer = 0, 0
    fn = CORRUPTIONS[corruption]
    with torch.no_grad():
        for image_tensors, labels in test_loader:
            image_tensors = fn(image_tensors, severity)
            image = image_tensors.cuda()
            preds = model(image).float()
            preds_size = torch.IntTensor([preds.size(1)] * image.size(0))
            preds = preds.permute(1, 0, 2).log_softmax(2)
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

            for pred, gt in zip(preds_str, labels):
                tot_ED += editdistance.eval(pred, gt)
                length_of_gt += len(gt)
            for pred, gt in zip(preds_str, labels):
                pred_w = utils.format_string_for_wer(pred).split(' ')
                gt_w = utils.format_string_for_wer(gt).split(' ')
                tot_ED_wer += editdistance.eval(pred_w, gt_w)
                length_of_gt_wer += len(gt_w)

    CER = tot_ED / length_of_gt
    WER = tot_ED_wer / length_of_gt_wer
    return CER, WER


if __name__ == '__main__':
    print(f"{'dataset':<6} {'corruption':<10} {'severity':>9} {'CER':>8} {'WER':>8}")
    for name in DATASETS:
        model, converter = load_model_and_converter(name)
        for corruption, sevs in SEVERITIES.items():
            for sev in sevs:
                cer, wer = evaluate(name, model, converter, corruption, sev)
                print(f"{name:<6} {corruption:<10} {sev:>9} {cer*100:7.2f}% {wer*100:7.2f}%")
