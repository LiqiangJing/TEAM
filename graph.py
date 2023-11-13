import pandas as pd
import re
import tqdm
import pickle
import hydra
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import functional, CrossEntropyLoss, MSELoss, Softmax, BCEWithLogitsLoss, MultiheadAttention
# from torch.distributions import Categorical, kl_divergence
import numpy as np
from torch import argmax
from collections import Counter
# from vit import ViT
from omegaconf import DictConfig, OmegaConf
import numpy as np
from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration, BartClassificationHead, \
    BartEncoder, BartAttention, BartForSequenceClassification, BartModel, shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BartTokenizer
from transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallConfig
from collections import deque

def main():
    map={}
    tkr = BartTokenizer.from_pretrained("/data/qiaoyang/ouyangkun/bart/bart-base")
    # model = Bart_Baseline(cfg, tkr)
    path_to_pkl='train_obj.pkl'
    path_to_data_df='train_df.tsv'
    pkl = open(path_to_pkl, 'rb')
    obj=pickle.load(pkl)

    lens=[]
    #统计token
    for tk in obj:
        lens.append(len(obj[tk]))
    lens.sort(reverse=True)
    print(lens)
    len_cnt=Counter(lens)
    print(len_cnt)
    exit()
    data = pd.read_csv(path_to_data_df, sep='\t', names=['pid', 'text', 'explanation'])
    with open('train_adj.pkl','wb') as f_save:
      for index in range(len(data)):
        row = data.iloc[index, :]
        src_text = row['text']
        pid_i=row['pid']
        src_list = re.split(' ', src_text)
        obj_start = len(src_list)
        # print(src_list)
        # print(len(src_list))
        obj_list = []
        cat_text = src_text
        if(pid_i in obj):
            for i in obj[str(pid_i)]:
                cat_text = cat_text + ' ' + i
                obj_list0 = re.split(' ', i)
                for t in obj_list0:
                    obj_list.append(t)
                # print(cat_text)
            cnt = obj_start + len(obj_list)
            adj = np.diag([1] *512)
            # print(obj_start)
            # print(adj)

            for i in range(obj_start-1):
                adj[i][i+1]=1
                adj[i+1][i]=1
                print(i)

            for x in range(obj_start):
                for y in range(len(obj_list)):
                    if(y + obj_start>512):
                        break
                    if (y % 2 == 1):
                        #print(src_list[x] + ' ' + obj_list[y])
                        if (src_list[x] == obj_list[y]):
                            adj[x][y + obj_start] = 1
                            adj[y + obj_start][x] = 1
                            # print('yes')
            # print(adj)
            idx = obj_start
            while (True):
                if (idx < cnt):
                    if(idx+1>=512):
                        break
                    adj[idx][idx + 1] = 1
                    adj[idx + 1][idx] = 1
                    idx = idx + 2
                else:
                    break

            map[pid_i]=adj

      pickle.dump(map, f_save)
    # f_save.close()
    # adj = open('Dataset/test_adj.pkl', 'rb')
    # adj = pickle.load(adj)
    # g=adj['720501577426018305']


if __name__ == '__main__':
    # adj = open('./Dataset/val_ADJ.pkl', 'rb')
    # adj = pickle.load(adj)
    # print(adj['726279551618273280'].shape)
    main()
