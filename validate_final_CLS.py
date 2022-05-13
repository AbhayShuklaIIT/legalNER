#!/usr/bin/env python
# coding: utf-8

import os
from utility import *
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from transformers import AdamW
from tqdm import tqdm


# def get_tokenizer():
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     new_tokens = ['<NE>', '</NE>']
#     special_tokens_dict = {'additional_special_tokens': new_tokens}
#     num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
#     return tokenizer


# class BertClassifier(nn.Module):

#     def __init__(self, device, dropout=0.5):

#         super(BertClassifier, self).__init__()

#         self.bert = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
#         self.tokenizer = get_tokenizer()
#         self.bert.resize_token_embeddings(len(self.tokenizer))
#         self.linear = nn.Linear(768*1, 11)
#         self.FL = nn.Softmax()
#         self.device = device

#     def forward(self, text):
#         encodings = self.tokenizer(text,padding='max_length', max_length = 256, truncation=True,return_tensors="pt")
#         encodings = encodings.to(self.device)
#         result, _ = self.bert(input_ids= encodings["input_ids"], attention_mask=encodings["attention_mask"],return_dict=False)

#         concat = result[:,0,:]
#         linear_output = self.linear(concat)
#         final_layer = self.FL(linear_output)
#         return final_layer


def get_list_from_labels_v1(roles, label_to_int_dict):
    y = []
    keys = list(label_to_int_dict.keys())
    for i in range(len(keys)):
        if keys[i] in roles:
            y.append(1)
        else:
            y.append(0)
    return y
def get_labels_from_ctx_v1(ne, contexts, label_to_int_dict, model, combined):
    int_to_label_dict = dict([(value, key) for key, value in label_to_int_dict.items()])
    probs = []
    model_labels = []
    if len(contexts) == 0:return []
    ip = []
    for ctx in contexts:
        ip.append(ne + "[SEP]" + ctx)
    if combined == 1:
        ip = []
        ip.append(ne + "[SEP]" + ". ".join(contexts))
    with torch.no_grad():
        p = model(ip)
#     print(np.array(p))
    a = torch.mean(p, 0)
    i_max = (torch.argmax(a))
    model_probs = [0 for _ in range(len(a))]
    model_probs[i_max]  = 1
    print(model_probs)
    return model_probs


def evaluate_post_v1(path, label_to_int_dict, model, combined, removed=[]):
    nes, ctx, roles, sl, fl = generate_training_examples(pd.read_excel(path, index_col = 0, na_filter = None), path)
    nes, ctx, roles, sl, fl = combine_varients(nes, ctx, roles, sl, fl, combined)
    roles = correct_roles(roles)
    yhat = []
    ytest = []
    for i in range(len(nes)):
        ne = nes[i]
        cs = ctx[i]
        rs = roles[i]
        
        flag = 0
        for r in removed:
            if r in rs:
                flag = 1
        if flag == 1: continue
        if len(rs) == 0:continue
        if len(cs) == 0:continue
        
        rs_ints = get_list_from_labels_v1(rs, label_to_int_dict)
        if fl[i] != "":
            pred_ints = get_labels_from_ctx_v1(ne, cs, label_to_int_dict, model, combined)
        else:
            pred_ints = [0 for i in range(len(label_to_int_dict.keys()))]
        yhat.append(pred_ints)
        ytest.append(rs_ints)
        if sum(rs_ints) == 0:
            print(rs)
            continue
    
    return ytest, yhat
def evaluate_v1(label_to_int_dict, model, combined):
    train = glob.glob("./train/*.xlsx")
    test = glob.glob("./test/*.xlsx")
    yhat = []
    ytest = []
    for i in tqdm(test):
        try:
            print(i)
            yt, yh = evaluate_post_v1(i, label_to_int_dict, model, combined)
            if len(yt) != len(yh):
                print(i, "not sam len")
                continue
            ytest.extend(yt)
            yhat.extend(yh)
        except Exception as e: 
#             print(e)
            print("ERROR s1", e)
            continue
#         break
    return ytest, yhat


def del_labels(ytest, yhat, label_to_int_dict):
    label_to_int_dict = get_label_to_int_dict()
    ytest_del = np.delete(ytest, 9, 1)
#     ytest_del = np.delete(ytest_del, 9, 1)
    yhat_del = np.delete(yhat, 9, 1)
#     yhat_del = np.delete(yhat_del, 9, 1)
    labels = list(label_to_int_dict.keys())
#     labels.remove('NOT APPLICABLE')
    labels.remove('OTHER')
#     print(yhat_del[2])
#     print(yhat[2])
#     print(list(label_to_int_dict.keys()))
#     print(labels)
    return ytest_del, yhat_del, labels

def print_cls_report_and_vals(ytest_f, yhat_f, labels, thres = 0.5):
    col = len(labels) +4
    yhat_ct = np.array(yhat_f) > thres
    print(classification_report(np.array(ytest_f), yhat_ct, target_names = labels))
    f1 = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[4])
    p = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[2])
    r = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[3])
    return p, r, f1


def print_result_v1(model_name, combined, device):
    label_to_int_dict = get_label_to_int_dict()
    model = torch.load( model_name, map_location=torch.device(device)) 
    ytest, yhat = evaluate_v1(label_to_int_dict, model, combined)
    ytest_n, yhat_n, labels = del_labels(ytest, yhat, label_to_int_dict)
    p, r, f = print_cls_report_and_vals(ytest_n, yhat_n, labels)
    return p, r, f

# print_result_v1('CLS_v4.pth', 0, "cuda")

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

from collections import defaultdict
def evaluate_doc_v2(path, label_to_int_dict, model, combined, removed = []):
    int_to_label_dict = dict([(value, key) for key, value in label_to_int_dict.items()])
    nes, ctx, roles, sl, fl = generate_training_examples(pd.read_excel(path, index_col = 0, na_filter = None), path)
    roles = correct_roles(roles)
    roles_to_ne_gs = defaultdict(set)
    for i in range(len(roles)):
        flag_rem = 0
        for rem in removed:
            if rem in roles[i]:
                flag_rem = 1
        if flag_rem == 1:
            continue
        for r in roles[i]:
            if r not in label_to_int_dict.keys():continue
            roles_to_ne_gs[r].add(nes[i])
    
#     print("Gold" , roles_to_ne_gs)
    roles_to_ne_pred = defaultdict(set)
    for i in range(len(nes)):
        if fl[i] == "":continue
        c_ri = []
        for ri in roles[i]:
            if ri in label_to_int_dict.keys():
                c_ri.append(ri)
                #                 print("ERROR - ", ri)
        
        if len(c_ri) == 0:continue
        if len(ctx[i]) == 0:continue
        j = ctx[i]
        
        pred = get_labels_from_ctx_v1(nes[i], j, label_to_int_dict, model, combined)
        
        pred = [pred]
        pred_ints = set()
        for p in pred:
            c = 0
            for p_i in p:
                if p_i == 1:
                    pred_ints.add(c)
                c = c + 1
            # pred_ints.append(torch.argmax(p))
        pred_ints = list(pred_ints)
        for ints in pred_ints:
            pred_label = int_to_label_dict[ints]
            roles_to_ne_pred[pred_label].add(nes[i])
    label_to_jaccard = {}
    for i in roles_to_ne_gs.keys():
        gs = roles_to_ne_gs[i]
        pred = roles_to_ne_pred[i]
        label_to_jaccard[i] = jaccard_similarity(gs,pred)
        # break
    
    for i in roles_to_ne_pred.keys():
        if i in roles_to_ne_gs.keys():continue
        label_to_jaccard[i] = 0

    return (label_to_jaccard)

from functools import reduce
  
def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

def eval_using_jaccard(model, label_to_int_dict, combined):
    train = glob.glob("./train/*.xlsx")
    test = glob.glob("./test/*.xlsx")
    labels_to_jaccards = defaultdict(list)
    c = 0;
    for i in tqdm(test):
        try:
            label_to_jaccard = evaluate_doc_v2(i, label_to_int_dict, model, combined)
#         except: 
        except Exception as e: 
            print(e)
            print("ERROR")
            continue
        for i in label_to_jaccard.keys():
            labels_to_jaccards[i].append(label_to_jaccard[i])
        # if c>2:break
        c = c + 1
    print()
    print(labels_to_jaccards)

    labels_to_jaccards_avg = defaultdict(list)
    for i in labels_to_jaccards.keys():
        labels_to_jaccards_avg[i] = Average(labels_to_jaccards[i])
    print(labels_to_jaccards_avg)
    return labels_to_jaccards_avg

def print_result_v2(model_name, combined, device):
    label_to_int_dict = get_label_to_int_dict()
    model = torch.load( model_name, map_location=torch.device(device))
    r = eval_using_jaccard(model, label_to_int_dict, combined)
#     print(r)
    avg_vec = []
    skip = ["NOT APPLICABLE", "OTHER"]
    for i in r.keys():
        print(i, r[i], sep = " - ")
        if i not in skip:
            avg_vec.append(r[i])
    print(sum(avg_vec)/len(avg_vec))
    return (sum(avg_vec)/len(avg_vec))