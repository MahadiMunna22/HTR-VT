import torch
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


def load_source(src_name):
    cfg = DATASETS[src_name]
    train_dataset = dataset.myLoadDS(cfg['train_ln'], cfg['data_path'], IMG_SIZE)
    model = HTR_VT.create_model(nb_cls=cfg['nb_cls'], img_size=IMG_SIZE[::-1])
    ckpt = torch.load(f"./output/{cfg['exp']}/best_CER.pth", map_location='cpu')
    model_dict = OrderedDict()
    for k, v in ckpt['state_dict_ema'].items():
        model_dict[k.replace('module.', '')] = v
    model.load_state_dict(model_dict, strict=True)
    model = model.cuda().eval()
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    src_charset = set(train_dataset.ralph.values())
    return model, converter, src_charset


def oov_rate(target_labels, src_charset):
    total, oov = 0, 0
    for lbl in target_labels:
        for ch in lbl:
            total += 1
            if ch not in src_charset:
                oov += 1
    return oov / total if total else 0.0


def cross_eval(src_name, tgt_name):
    model, converter, src_charset = load_source(src_name)
    cfg_t = DATASETS[tgt_name]
    test_dataset = dataset.myLoadDS(cfg_t['test_ln'], cfg_t['data_path'], IMG_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,
                                               pin_memory=True, num_workers=4)

    tot_ED, length_of_gt = 0, 0
    tot_ED_wer, length_of_gt_wer = 0, 0
    with torch.no_grad():
        for image_tensors, labels in test_loader:
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
    oov = oov_rate(test_dataset.tlbls, src_charset)
    return CER, WER, oov


if __name__ == '__main__':
    pairs = [
        ('LAM', 'IAM'), ('LAM', 'READ'),
        ('IAM', 'LAM'), ('IAM', 'READ'),
        ('READ', 'LAM'), ('READ', 'IAM'),
    ]
    print(f"{'source':<6} {'target':<6} {'CER':>8} {'WER':>8} {'tgt_char_OOV%':>14}")
    for src, tgt in pairs:
        cer, wer, oov = cross_eval(src, tgt)
        print(f"{src:<6} {tgt:<6} {cer*100:7.2f}% {wer*100:7.2f}% {oov*100:13.2f}%")
