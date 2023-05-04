import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import functional, CrossEntropyLoss, MSELoss, Softmax, BCEWithLogitsLoss, MultiheadAttention
# from torch.distributions import Categorical, kl_divergence

from torch import argmax
import torch.nn.functional as F
from vit import ViT

import numpy as np
# import cudamat as cm
#
# cm.cublas_init()
from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration, BartClassificationHead, \
    BartEncoder, BartAttention, BartForSequenceClassification, BartModel, shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BartTokenizer
from transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallConfig

from utils import _init_weights

from collections import deque
from torch.nn import MultiheadAttention
#QKV,
#Q=t*w K=W V=W
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers.models.bart.modeling_bart import *


class Multihead_Attention(nn.Module):
    """
    Multi-head Attention
    """

    def __init__(self,
                 model_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 bias: bool = False, ):
        """
        initialization for variables and functions
        :param model_dim: hidden size
        :param num_heads: head number, default 8
        :param dropout: dropout probability
        """
        super(Multihead_Attention, self).__init__()

        self.head_dim = model_dim // num_heads
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.linear_keys = nn.Parameter(torch.eye(768))#(model_dim, num_heads * self.head_dim, bias=bias)
        self.linear_values = nn.Parameter(torch.eye(768))#nn.linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.linear_query = nn.Parameter(torch.eye(768))#nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.sigmoid = nn.Hardtanh(min_val=0)

    def forward(self, hidden_states, key_value_states, return_top_attn=False):
        """
        run multi-head attention
        :param key: key, [batch, len, size]
        :param value: value, [batch, len, size]
        :param query: query, [batch, len, size]
        :param mask: mask
        :param layer_cache: layer cache for transformer decoder
        :param type: "self" or "context"
        :param tau: temperature, will be deprecated
        :param Bernoulli: use Bernoulli selection or not
        :return: attention output and attention weights
        """
        query = hidden_states
        key = value = key_value_states

        batch_size = key.size(0)
        head_dim = self.head_dim
        head_count = self.num_heads
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, head_dim) \
                .transpose(1, 2)  # [batch, head, len, head_dim]

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * head_dim)  # [batch, len, size]

        # For transformer decoder.
        # denote the device for multi-gpus
        query, key, value = torch.matmul(query,self.linear_query), \
                            torch.matmul(key,self.linear_keys), \
                            torch.matmul(value, self.linear_values)  # [batch, len, size]
        key = shape(key)  # [batch, head, k_len, head_dim]
        value = shape(value)  # [batch, head, v_len, head_dim]

        query = shape(query)  # [batch, head, q_len, head_dim]

        key_len = key.size(2)
        query_len = query.size(2)

        query = query / math.sqrt(head_dim)

        scores = torch.matmul(query, key.transpose(2, 3))  # [batch, head, q_len, k_len]

        # use Bernoulli selection or not
        attn = self.softmax(scores)  # [batch, head, q_len, k_len]

        drop_attn = self.dropout(attn)  # [batch, head, q_len, k_len]
        context = unshape(torch.matmul(drop_attn, value))  # [batch, q_len, size]

        # output = self.final_linear(context)  # [batch, q_len, size]
        #
        # top_attn = attn \
        #                .view(batch_size, head_count,
        #                      query_len, key_len)[:, 0, :, :] \
        #     .contiguous()  # [batch, q_len, k_len]
        # if return_top_attn:
        #     return output, top_attn
        return context


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.eye(768))#torch.FloatTensor(in_features, out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        # print(text.shape,self.weight.shape)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1

        # print(denom)
        # print(adj.shape, hidden.float().shape)
        output = torch.matmul(adj.float(), hidden.float()) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class Bart_Baseline(nn.Module):

    def __init__(self, cfg, tkr):
        super(Bart_Baseline, self).__init__()
        self.cross_modal_encoder = MultiheadAttention(embed_dim=768, num_heads=16, kdim=768, vdim=768)
        self.cfg = cfg
        # self.tkr = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
        self.tkr = tkr
        self.Multihead_Attention=Multihead_Attention(768)
        # self.model_gen = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
        self.model_gen = BartForConditionalGeneration.from_pretrained("./bart-base")
        self.img_encoder = ViT(image_size=224, patch_size=32, num_classes=10, dim=768, depth=1, heads=16, mlp_dim=2048)

        self.hsize = self.model_gen.config.hidden_size
        self.gc = GraphConvolution(768,768)#train.batch_size



        # self.gru = nn.GRU(input_size=self.hsize, hidden_size=self.hsize, batch_first=True)
        # self.strat_emb = nn.Embedding(9, self.hsize, padding_idx=0)

        # self.cross_modal_encoder = MultiheadAttention(embed_dim=512, num_heads=16)

        # self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # self.sgm = nn.Sigmoid()
        # self.bceloss = nn.BCELoss(reduction='none')  # no sigmoid layer inside
        # self.celoss = nn.CrossEntropyLoss()
        # self.proj_pred_cls = nn.Sequential(nn.Linear(self.hsize, 256),
        #                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #                                   nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #                                    nn.Linear(64, 8))
        # self.proj_cls = nn.Sequential(nn.Linear(self.hsize, 256),
        #                               nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #                               nn.Linear(256, 64),
        #                  nn.Linear(256, 64),
        #         #                                                  nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #                               nn.Linear(64, 8))
        # self.strat_freat_dic = {i: [] for i in range(7)}
        # self.img_encoder = ViT(image_size=224, patch_size=32, num_classes=10, dim=768, depth=1, heads=16, mlp_dim=2048)
        # self.proj_q = nn.Linear(self.hsize, 768)
        # self.proj_for_cat = nn.Linear(768, self.hsize)

    def forward(self, input_image, input_ids, graph,text,attention_mask, target_ids,
                mode='train',
                **kwargs):
        device = input_ids.device
        bs = input_ids.shape[0]

        # print(input_image.device)
        # print(model.device)
        # print(input_ids)
        # print(input_ids.shape)
        # image_feat = self.img_encoder(input_image)
        # img_mask = torch.ones([bs, image_feat.shape[1]], device=device, dtype=torch.long)

        input_embed = self.model_gen.get_input_embeddings()(input_ids)
        # print(image_feat.shape)
        # print(image_feat.dim)
        # print(input_embed.shape)
        # concat_feat = torch.cat([image_feat, input_embed], dim=1)
        # concat_msk = torch.cat([img_mask, attention_mask], dim=1)
        concat_feat = input_embed
        concat_msk = attention_mask

        label=target_ids

        context_enc_out = self.model_gen.get_encoder()(inputs_embeds=concat_feat, attention_mask=concat_msk)

        context_enc_out_feat = context_enc_out.last_hidden_state
        #print(context_enc_out_feat.shape)

       # input_ids.dtype = "int32"
       # graph.dtype="int32"
       #  print(graph.shape)
       #  print(type(graph))
       #  exit()
        x = self.gc(context_enc_out_feat, graph)
        # qx=x.transpose(1,0)
        # kv=context_enc_out_feat.transpose(1,0)
        # cross_modal_feat=torch.matmul(context_enc_out_feat,x.transpose(2,1))
        #cross_modal_feat = self.Multihead_Attention(x, context_enc_out_feat)#[0].transpose(1, 0)
        # print(x)
        # print(cross_modal_feat)
        # exit()
        # print(cross_modal_feat.shape)
        # exit()
        # x = F.relu(self.gc2(x, graph))
        gen_feat =context_enc_out_feat+x#torch.matmul(cross_modal_feat,context_enc_out_feat)+context_enc_out_feat#context_enc_out_feat*cross_modal_feat.transpose(1,0)#x+context_enc_out_feat #context_enc_out_feat#context_enc_out_feat #
        gen_mask = concat_msk

        if mode == 'train':

            enc_output = BaseModelOutput(last_hidden_state=gen_feat)

            gen = self.model_gen(encoder_outputs=enc_output, attention_mask=gen_mask,labels=label)

            loss = [gen.loss]

            return loss

        elif mode == 'eval' or mode == 'gen':
            with torch.no_grad():

                enc_output = BaseModelOutput(last_hidden_state=gen_feat)

                if mode == 'eval':
                    # 计算 ppl
                    # gen = self.model_gen(encoder_outputs=enc_output, attention_mask=gen_mask, labels=label)
                    # lm_logits = gen.logits
                    # loss_fct = CrossEntropyLoss(reduction='none')
                    # masked_lm_loss = loss_fct(lm_logits.view(-1, len(self.tkr)) - 8, label.view(-1))
                    # batch_masked_lm_loss = masked_lm_loss.view(bs, -1)
                    # label_size = torch.sum(label.ne(-100), dim=1).type_as(batch_masked_lm_loss)
                    # ppl_value = torch.exp(
                    #     torch.mean(torch.sum(batch_masked_lm_loss, dim=1).float() / label_size.float()))
                    #
                    # return ppl_value

                    # 计算eval 集 loss
                    gen = self.model_gen(encoder_outputs=enc_output, attention_mask=gen_mask, labels=label)
                    return gen.loss

                elif mode == 'gen':

                    generation_cfgs = {"max_length": self.cfg.eval.eval_max_len,
                                       "min_length": self.cfg.eval.eval_min_len,
                                       "pad_token_id": self.tkr.pad_token_id,
                                       'eos_token_id': self.tkr.eos_token_id, "num_beams": self.cfg.eval.num_beams,
                                       'top_p': self.cfg.eval.top_p, 'top_k': self.cfg.eval.top_k,
                                       'temperature': self.cfg.eval.temperature, 'do_sample': True,
                                       'repetition_penalty': self.cfg.eval.repetition_penalty,
                                       'no_repeat_ngram_size': self.cfg.eval.no_repeat_ngram_size}
#原来生成结果
                    gen_result = self.model_gen.generate(encoder_outputs=enc_output,
                                                         attention_mask=gen_mask,
                                                         **generation_cfgs)
#修改后
                    # gen_result = self.model_gen.generate(encoder_outputs=enc_output,
                    #                                      attention_mask=gen_mask,
                    #                                      )#不给予其他参数

                    gen_decoded = self.tkr.batch_decode(gen_result, skip_special_tokens=True)

                    return gen_decoded

        else:
            raise ValueError('Mode should be among [train, eval, gen].')
