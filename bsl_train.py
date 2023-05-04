import os
import time
import inspect
import math
import sys
from tqdm import tqdm
from pprint import pformat
import pickle
# import inputters
import ast
from functools import partial

import torch
# import torchtext.data
import torch.nn as nn
import torch.optim as optim

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
# from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bsl_model import Bart_Baseline
from torchvision import transforms, models
import bsl_model as model_file

from torch.utils.data import DataLoader

from dataset import MineDataset

from utils import setup_seed, writr_gt, send_to_device, make_exp_dirs

from metrics_eval import eval_metrics
from log_utils import setting_logger

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import BartTokenizer
from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration, BartClassificationHead, \
    BartEncoder, BartAttention, BartForSequenceClassification, BartModel, shift_tokens_right

import nltk
# log = Logger()

import logging
def get_pretrained_model(model, saved_dir, log):
    saved_models = os.listdir(saved_dir)
    if len(saved_models) != 0:
        saved_models.sort()
        from_ep = saved_models[-1][5] + saved_models[-1][6] + saved_models[-1][7]
        saved_model_path = os.path.join(saved_dir, saved_models[-1])
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict)
        log.info('Load state dict from %s' % str(saved_model_path))
    else:
        from_ep = -1
        log.info('Initialized randomly (with seed)')
    return model, int(from_ep)


def load_data(tkr, cfg):
    train_file = 'Dataset/train_df.tsv'
    val_file = 'Dataset/val_df.tsv'
    test_file = 'Dataset/test_df.tsv'
    train_pkl = 'Dataset/train_obj.pkl'
    val_pkl = 'Dataset/valid_obj.pkl'
    test_pkl = 'Dataset/test_obj.pkl'
    train_adj = 'Dataset/train_ADJ.pkl'
    val_adj = 'Dataset/val_ADJ.pkl'
    test_adj = 'Dataset/test_ADJ.pkl'
    train_tsv_dict='Dataset/train_tsv_concept_dict.pkl'
    val_tsv_dict='Dataset/val_tsv_concept_dict.pkl'
    test_tsv_dict='Dataset/test_tsv_concept_dict.pkl'
    train_obj_dict = 'Dataset/train_obj_concept_dict.pkl'
    val_obj_dict = 'Dataset/valid_obj_concept_dict.pkl'
    test_obj_dict = 'Dataset/test_obj_concept_dict.pkl'
   # path_to_concept='Dataset/assertions.csv'
    img_dir = "Dataset/images"
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MineDataset(train_file,train_pkl, train_adj,train_tsv_dict,train_obj_dict,img_dir, tkr, image_transform,cfg)
    val_dataset = MineDataset(val_file, val_pkl,val_adj,val_tsv_dict,val_obj_dict,img_dir, tkr, image_transform,cfg)
    test_dataset = MineDataset(test_file,test_pkl,test_adj, test_tsv_dict,test_obj_dict,img_dir, tkr, image_transform,cfg)
    # train_dataset = MineDataset(train_file, tkr, cfg)
    # val_dataset = MineDataset(val_file, tkr, cfg)
    # test_dataset = MineDataset(test_file, tkr, cfg)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=8, shuffle=True)
                                #collate_fn=partial(train_dataset, toker=tkr, cfg=cfg))
    eval_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, num_workers=8, shuffle=False)
                                #collate_fn=partial(val_dataset, toker=tkr, cfg=cfg))
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=8, shuffle=False)
                                #collate_fn=partial(test_dataset, toker=tkr, cfg=cfg))#cfg.eval.batch_size

    return train_dataloader, eval_dataloader, test_dataloader


def eval_net(ep, model, loader, log_path, device, log):
    ppl_mean = 0
    model.eval()

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            batch = send_to_device(batch, device)
            ppl = model(**batch, mode='eval')
            ppl_mean += ppl.cpu().numpy()

    ppl_mean = ppl_mean / idx

    return ppl_mean


def gen_net(ep, model, loader, log_path, device, log):
    log_txt_name = os.path.join(log_path, f'gen_{ep}.txt')
    log_txt = open(log_txt_name, 'w')
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            batch = send_to_device(batch, device)
            query_infer = model(**batch, mode='gen')
            log_txt.write('\n'.join(query_infer) + '\n')

    log_txt.flush()
    log_txt.close()
    gt_name = os.path.join(log_path, 'gt4test.txt')
    scores = eval_metrics(log_txt_name, gt_name)

    return scores


def load_model(model, save_path, epoch):
    model_pth_path = os.path.join(save_path, 'epoch' + str(epoch).zfill(3) + '.pth')
    state_dict = torch.load(model_pth_path)
    model.load_state_dict(state_dict)
    return model


def save_model(accelerator, model, save_path, epoch):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(),
                     os.path.join(save_path, 'epoch' + str(epoch).zfill(3) + '.pth'))


def run_stage(cfg, model, lr_sche, opt,
              train_loader, eval_loader, test_loader,
              log_path, device,
              accelerator, log):
    print_every = int(len(train_loader) / 10)
    eval_every = 1
    save_every = 1000

    max_epoch = cfg.train.max_epoch

    res_metric_key = ['B1', 'B2', "B3", 'B4', 'RL', 'Cdr', 'Mtr']
    best_res_dic = {i: 0.0 for i in res_metric_key}

    best_ppl = 1e6
    best_ppl_ep = 0
    scores = []
    for epoch in range(max_epoch):
        model.train()
        log.info(f"{'-' * 20} Current Epoch:  {epoch} {'-' * 20}")

        time_now = time.time()
        show_loss = 0

        for idx, batch in enumerate(train_loader):

            opt.zero_grad()

            batch = send_to_device(batch, device)
            loss = model(**batch)
            # if len(loss) >1:
            #     loss_mean = (loss[0] + cfg.train.lam1*loss[1] + cfg.train.lam2*loss[2])
            # else:
            #     loss_mean = sum(loss)
            loss_mean = sum(loss)
            accelerator.backward(loss_mean)
            opt.step()

            cur_lr = opt.param_groups[-1]['lr']
            show_loss += loss_mean.detach().cpu().numpy()
            # print statistics
            if idx % print_every == print_every - 1 and accelerator.is_main_process:
                cost_time = time.time() - time_now
                time_now = time.time()
                log.info(
                    f'lr: {cur_lr:.6f} | step: {idx + 1}/{len(train_loader) + 1} | time cost {cost_time:.2f}s | loss: {(show_loss / print_every):.4f}')
                show_loss = 0

            lr_sche.step()

        log.info(f'current lr:  {cur_lr}')

        if (epoch % eval_every) == (eval_every - 1) and epoch >= 0:

            log.info('Evaluating Net...')
            ppl = eval_net(epoch, model, eval_loader, log_path, device, log)
            if ppl <= best_ppl:
                best_ppl = ppl
                best_ppl_ep = epoch
            log.info(f"Cur epoch: {epoch} | PPL: {ppl} | Best_ppl_ep: {best_ppl_ep} | Best ppl: {best_ppl}")

            ## save model
            save_model(accelerator, model, log_path, epoch)
            log.info('Model Saved! ')
            # 对每个epoch的模型eval
        model = load_model(model, log_path, epoch)
        log.info(f'Model Loaded! Best ep: {epoch}')
        ppl = eval_net(epoch, model, test_loader, log_path, device, log)
        score = gen_net(epoch, model, test_loader, log_path, device, log)
        score['PPL'] = ppl
        print(score)
        scores.append(score)
    return scores

    ## generate and calculate metrics
    # model = load_model(model, log_path, best_ppl_ep)
    # log.info(f'Model Loaded! Best ep: {best_ppl_ep}')
    # ppl = eval_net(best_ppl_ep, model, test_loader, log_path, device, log)
    # scores = gen_net(best_ppl_ep, model, test_loader, log_path, device, log)
    # scores['PPL'] = ppl






@hydra.main(config_path="conf", config_name="basic_cfg",version_base='1.2.0')
def main(cfg: DictConfig):
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    # accelerator = Accelerator()
    #device = accelerator.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_seed(int(cfg.train.seed))

    log_path = make_exp_dirs(cfg.name)

    # log_file_path = os.path.join(os.path.abspath(log_path),'details.log')
    # logging.basicConfig(filename=log_file_path, filemode = "w",level=logging.DEBUG,
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # log = logging.getLogger()

    global log
    log = setting_logger(log_path)

    # tkr = BartTokenizer.from_pretrained("/home/matongqing/bart/bart-base")
    tkr = BartTokenizer.from_pretrained("./bart-base")
    model = Bart_Baseline(cfg, tkr)
    from_scratch_params = list(map(id, nn.ModuleList([model.gc]).parameters()))
    other_params = filter(lambda p: id(p) not in from_scratch_params, model.parameters())
    optimizer = AdamW([
        {'params': other_params, 'lr': cfg.train.pt_lr},
        {'params': nn.ModuleList([model.gc]).parameters(), 'lr': cfg.train.lr}],
        weight_decay=cfg.train.weight_decay)
    #optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    train_dataloader, eval_dataloader, test_dataloader = load_data(tkr, cfg)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=int(cfg.train.max_epoch) * len(train_dataloader))

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # log model file into log
    log.info(inspect.getsource(model_file))

    log.info(f'Found device: {device}')

    str_cfg = OmegaConf.to_yaml(cfg)
    log.info(f"Config: {str_cfg}")

    log.info(f"train data: {cfg.train.batch_size * len(train_dataloader)}")
    log.info(f"eval data: {cfg.train.batch_size * len(eval_dataloader)}")
    log.info(f"test data: {cfg.eval.batch_size * len(test_dataloader)}")

    writr_gt(test_dataloader, log_path, tkr=tkr)
#调试
    if cfg.debug:
        cfg.train.max_epoch = 2

    scores = run_stage(cfg=cfg, model=model,
                       lr_sche=lr_scheduler, opt=optimizer,
                       train_loader=train_dataloader, eval_loader=eval_dataloader, test_loader=test_dataloader,
                       log_path=log_path, device=device, accelerator=accelerator, log=log)

    log.info(f"Config: {str_cfg}")
    log.info(pformat(scores))


if __name__ == '__main__':
    main()
