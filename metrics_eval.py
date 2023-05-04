from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
import nltk
from rouge_score import rouge_scorer
# from sentence_transformers import SentenceTransformer
# from bert_score import score as b_score
# from torchmetrics.text.bert import BERTScore
from pprint import pprint
import torch

def open_file(hyp_file, gt_file):
    with open(hyp_file, 'r', encoding='utf-8') as r1, open(gt_file, 'r', encoding='utf-8') as r2:
        for line1, line2 in zip(r1, r2):
            a, b, c = 'empty', line1.strip(), line2.strip()
            yield a, b, c


def eval_metrics(hyp_txt, ref_txt):
    # calculate metrices
    with open(ref_txt, 'r') as gtr:
        gt_list = gtr.readlines()
    with open(hyp_txt, 'r') as genr:
        gen_list = genr.readlines()

    for_eval_gt_dic = {}
    for_eval_gen_dic = {}
    for_dist_hyper = []

    for i in range(len(gen_list)):
        # ref1 = gt_list[i].replace('\n','')
        # gen1 = gen_list[i].replace('\n','')
        # for_dist_hyper.append(gen_list[i].replace('\n','').split(' '))

        # ref = ' '.join(nltk.word_tokenize(gt_list[i]))
        # gen = ' '.join(nltk.word_tokenize(gen_list[i]))
        # for_dist_hyper.append(nltk.word_tokenize(gen_list[i]))

        ref = ' '.join(nltk.word_tokenize(gt_list[i].lower()))
        gen = ' '.join(nltk.word_tokenize(gen_list[i].lower()))
        for_dist_hyper.append(nltk.word_tokenize(gen_list[i].lower()))

        for_eval_gt_dic[i] = [ref]
        for_eval_gen_dic[i] = [gen]

    scores = eval_metrics_str(for_eval_gen_dic, for_eval_gt_dic)

    dist_dic = calc_distinct(for_dist_hyper)
    scores.update(dist_dic)

    for k in scores:
        scores[k] = f"{scores[k] * 100:.3f}"
    return scores

    return scores

    # result_dic = {}
    # lines = [[item_id, {'0': pred}, {'0': gt}] for item_id, pred, gt in open_file(hyp_txt, ref_txt)]

    # item_ids, preds, gts = list(zip(*lines))

    # hpys_list =  [' '.join(list(val.values())) for i, val in enumerate(preds)]
    # preds = {i: list(val.values()) for i, val in enumerate(preds)}
    # gts = {i: list(val.values()) for i, val in enumerate(gts)}

    # b = Bleu()
    # score, _ = b.compute_score(gts=gts, res=preds)
    # b1, b2, b3, b4 = score

    # r = Rouge()
    # score, _ = r.compute_score(gts=gts, res=preds)
    # rl = score

    # m = Meteor()
    # score, _ = m.compute_score(gts=gts, res=preds)
    # mtr = score

    # c = Cider()
    # score, _ = c.compute_score(gts=gts, res=preds)
    # cdr = score

    # dist = calc_distinct(hpys_list)

    # result_dic['Bleu_1'] = b1
    # result_dic['Bleu_2'] = b2
    # result_dic['Bleu_3'] = b3
    # result_dic['Bleu_4'] = b4
    # result_dic['Rouge_L'] = rl
    # result_dic['METEOR'] = mtr
    # result_dic['CIDEr'] = cdr
    # result_dic.update(dist)
    # for k in result_dic:
    #     result_dic[k] = f"{result_dic[k]*100:.3f}"
    # return result_dic


# format: {0:['this is a good day!'],1:['dsgg sdg']}

def eval_metrics_str(hyp_dic, ref_dic):
    hyp_list=[]
    ref_list=[]
    for i in hyp_dic.values():
        hyp_list.append(i)
    for i in ref_dic.values():
        ref_list.append(i)
    result_dic = {}
    b = Bleu()
    score, _ = b.compute_score(gts=ref_dic, res=hyp_dic)
    b1, b2, b3, b4 = score

    r = Rouge()
    score, _ = r.compute_score(gts=ref_dic, res=hyp_dic)

    rl = score
    count=0
    rouge1=0
    rouge2=0
    met=0
    for reference, hypothesis in zip(ref_list, hyp_list):
        reference=str(reference).strip('[]\'')
        hypothesis=str(hypothesis).strip('[]\'')
        count += 1
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        met += nltk.translate.meteor_score.meteor_score([reference], hypothesis)
        scores = scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
    rouge1 = rouge1 / count
    rouge2 = rouge2 / count
    met = met / count
    # m = Meteor()
    # score, _ = m.compute_score(gts=ref_dic, res=hyp_dic)
    # mtr = score

    c = Cider()
    score, _ = c.compute_score(gts=ref_dic, res=hyp_dic)
    cdr = score
    # bertscore = BERTScore()
    # P,R,F1= b_score(hyp_list, ref_list, lang="en", verbose=True)
    # rounded_score = {k: [round(v, 3) for v in vv] for k, vv in bert_s.items()}
    # pprint(bert_s)

    # Cosine Similarity for Sentence BERT representation
    # device = torch.device('cpu')
    # sentence_transformer_model = SentenceTransformer('./bart-base')
    # sentence_transformer_model.to(device)
    # sentence_embeddings = sentence_transformer_model.encode(ref_list)
    # sentence_embeddings2 = sentence_transformer_model.encode(hyp_list)
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # output = torch.mean(cos(torch.tensor(sentence_embeddings), torch.tensor(sentence_embeddings2))).item()
    # print("Cosine Similarity of Sentence Representation using S-BERT:", output)


    result_dic['Bleu_1'] = b1
    result_dic['Bleu_2'] = b2
    result_dic['Bleu_3'] = b3
    result_dic['Bleu_4'] = b4
    result_dic['Rouge_L'] = rl
    result_dic['Rouge1'] = rouge1
    result_dic['Rouge2'] = rouge2
    result_dic['METEOR'] = met
    # result_dic['sent'] = output
    # result_dic['P'] = P
    # result_dic['R'] = R
    # result_dic['F1'] = F1
    # result_dic['CIDEr'] = cdr
    return result_dic


# hyps: ['sadsa,dsff','asd dfdsf']
def calc_distinct(hyps):
    dist_dic = {}
    for k in range(1, 3):
        d = {}
        tot = 0
        for sen in hyps:
            for i in range(0, len(sen) - k):
                key = tuple(sen[i:i + k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn('the distinct is invalid')
            dist = 0.
        dist_dic[f'Dist-{k}'] = dist
    return dist_dic


if __name__ == '__main__':
    t1 = 'gen_18.txt'
    t2 = 'gt4test.txt'
    RES=eval_metrics(t1, t2)
    print(RES)