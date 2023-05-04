import os
import time
import torch
import numpy as np
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import MineDataset
import os
import time
import inspect
import math
import sys
from tqdm import tqdm
import pandas as pd
import torch
# import torchtext.data
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torchvision import transforms, models

from torch.utils.data import DataLoader

from dataset import MineDataset

from transformers import BartTokenizer


# from flashtext import KeywordProcessor
# from eval.caption import cal_score_from_txt

def make_exp_dirs(exp_name):
    day_logs_root = 'generation_logs/' + time.strftime("%Y-%m%d", time.localtime())
    os.makedirs(day_logs_root, exist_ok=True)
    exp_log_path = os.path.join(day_logs_root, exp_name)

    # model_save_root ='saved_models/'
    # model_save_path = os.path.join(model_save_root, exp_name)

    os.makedirs(exp_log_path, exist_ok=True)  # log dir make
    # os.makedirs(model_save_path, exist_ok=True)  # model save dir make

    return exp_log_path


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def writr_gt(test_dataloader, log_dir, tkr):
    gt_file_name_test = os.path.join(log_dir, ('gt4test.txt'))
    gt_txt_test = open(gt_file_name_test, 'w')

    gt_with_id_file_name_test = os.path.join(log_dir, ('gt_test.txt'))
    gt_with_id_txt_test = open(gt_with_id_file_name_test, 'w')

    for idx, test_data in tqdm(enumerate(test_dataloader)):
        for i in range(len(test_data['input_ids'])):
            # print(test_data['input_ids'])

            context = tkr.decode(test_data['input_ids'][i], skip_special_tokens=True)

            label_pad = test_data['target_ids'][i].masked_fill(test_data['target_ids'][i] == -100, 0)

            label = tkr.decode(label_pad, skip_special_tokens=True)
            # strat = tkr.decode(test_data['strat'][i], skip_special_tokens=True)

            # gt_with_id_txt_test.write(f"{strat} \t {noun} \t {context} \t {label}  \n")
            gt_with_id_txt_test.write(f"{context} \t\n")
            gt_txt_test.write(label + '\n')

    for txt in [gt_txt_test, gt_with_id_txt_test]:
        txt.flush()
        txt.close()


@hydra.main(config_path="conf", config_name="basic_cfg", version_base='1.2.0')
def main(cfg: DictConfig):
    test_file = 'Dataset/val_df.tsv'
    data = pd.read_csv(test_file, sep='\t', names=['pid', 'text', 'explanation'])
    print(data['text'])
    img_dir = "Dataset/images"
    tkr = BartTokenizer.from_pretrained("./bart-base")
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_pkl = 'Dataset/valid_obj.pkl'
    test_obj_dict = 'Dataset/valid_obj_concept_dict.pkl'
    test_tsv_dict = 'Dataset/val_tsv_concept_dict.pkl'
    test_adj = 'Dataset/val_ADJ.pkl'
    test_dataset = MineDataset(test_file, test_pkl, test_adj, test_tsv_dict, test_obj_dict, img_dir, tkr,
                               image_transform, cfg)
    test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=8, shuffle=True)
    log_path = '/data/qiaoyang/bart/generation_logs'
    writr_gt(test_dataloader, log_path, tkr=tkr)


if __name__ == '__main__':
    main()