import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models

from transformers import BartTokenizer, BartForConditionalGeneration, BartModel, \
    AdamW, BartConfig, BartPretrainedModel
import re
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pickle


def read_single_csv(input_path):
    df_chunk = pd.read_csv(input_path, chunksize=1000, delimiter='\t')
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df = pd.concat(res_chunk)
    return res_df


class MineDataset(Dataset):
    def __init__(self, path_to_data_df, path_to_obj, path_to_adj, path_to_tsv_dict, path_to_obj_dict, path_to_images,
                 tokenizer, image_transform, cfg):
        # 抽取的object输入部分
        self.data = pd.read_csv(path_to_data_df, sep='\t', names=['pid', 'text', 'explanation'])
        obj = open(path_to_obj, 'rb')
        adj = open(path_to_adj, 'rb')
        tsv_dict = open(path_to_tsv_dict, 'rb')
        obj_dict = open(path_to_obj_dict, 'rb')
        # 数据库抽取的信息词的部分
        # self.cpt=pd.read_csv(path_to_pkl,delimiter='\t')
        self.adj = pickle.load(adj)
        self.obj = pickle.load(obj)
        self.tsv_dict = pickle.load(tsv_dict)
        self.obj_dict = pickle.load(obj_dict)
        self.path_to_images = path_to_images
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.cfg = cfg

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]

        pid_i = row['pid']
        src_text =row['text']
        tsv_words = src_text.split(' ')

        obj_len = 0
        adj_i = self.adj[str(pid_i)]
        if (pid_i in self.obj):
            obj_len = len(self.obj[str(pid_i)])
            #move obj
            for i in self.obj[str(pid_i)]:
                src_text = src_text + ' ' + i
        if pid_i in self.tsv_dict.keys():
            tsv_dict = self.tsv_dict[str(pid_i)]
            for tsv_word in tsv_words:
                tsv_word = tsv_word.strip('{},.\'!?#*')
                if tsv_word in tsv_dict:
                    src_text = src_text + ' ' + tsv_dict[tsv_word]
        if (pid_i in self.obj):
            obj_list = self.obj[str(pid_i)]
            for obj_words in obj_list:
                obj_word = obj_words.split(' ')
                for word in obj_word:
                    if word in self.obj_dict.keys():
                        lists = self.obj_dict[word]
                        src_text = src_text + ' ' + lists[0]
        # print(src_text)
        target_text = row['explanation']

        max_length = self.cfg.dataset.max_len
        encoded_dict = self.tokenizer(
            src_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            add_prefix_space=True
        )
        src_ids = encoded_dict['input_ids'][0]
        src_mask = encoded_dict['attention_mask'][0]

        image_path = os.path.join(self.path_to_images, pid_i + '.jpg')
        img = np.array(Image.open(image_path).convert('RGB'))
        img_inp = self.image_transform(img)

        encoded_dict = self.tokenizer(
            target_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            add_prefix_space=True
        )

        target_ids = encoded_dict['input_ids'][0]
        sample = {
            "input_ids": src_ids,
            "attention_mask": src_mask,
            "input_image": img_inp,
            "target_ids": target_ids,
            "text": src_text,
            "graph": adj_i,
            "obj_len": obj_len
        }
        return sample

    def __len__(self):
        return self.data.shape[0]
    # array = re.split('[ ,.]', src_text)
    # self.cpt.columns = ['uri', 'relation', 'start', 'end', 'json']
    # num=len(self.cpt)
    # for words in array:
    #     for cnt in range(num):
    #        item=self.cpt.iloc[cnt]
    #        if (str(words) in str(item['start']) and '/c/en/' in str(item['end']) ):
    #         src_text=src_text+','+str(self.cpt['end']).lstrip('/cen')
    # print(src_text)

    # print(adj_i.shape)
    #  print(self.path_to_adj)