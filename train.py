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

label_to_int_dict = get_label_to_int_dict()

def get_train_data(path):
    df = pd.read_pickle(path)
    x_train = df["x_train"]
    y_train = df["y_train"]
    y_train = list(y_train)
    df_train = pd.DataFrame.from_dict({"texts":x_train, "labels":y_train})
    df_train, df_val = np.split(df_train, [int(1*len(df_train))])
    print(len(df_train),len(df_val))
    return df_train, df_val, x_train, y_train



def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    new_tokens = ['<NE>', '</NE>']
    special_tokens_dict = {'additional_special_tokens': new_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = list(df['labels'])
        self.texts = list(df['texts'])

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, device, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.tokenizer = get_tokenizer()
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(768*1, 10)
        self.FL = nn.Softmax()
        self.device = device

    def forward(self, text):
        encodings = self.tokenizer(text,padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
        encodings = encodings.to(self.device)
        result, _ = self.bert(input_ids= encodings["input_ids"], attention_mask=encodings["attention_mask"],return_dict=False)

        concat = result[:,0,:]
        linear_output = self.linear(concat)
        final_layer = self.FL(linear_output)
        return final_layer


def get_loss_weights(y_train):
    wt = (np.sum(np.array(y_train), axis = 0) / np.sum(np.array(y_train)))
    wt = np.reciprocal(wt)
    weights = torch.tensor(wt)
    return weights


def get_optimizer(model):
    pretrained = model.bert.parameters()
    pretrained_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]
    new_params= [v for k, v in model.named_parameters() if k not in pretrained_names]
    optimizer = AdamW(
        [{'params': pretrained, 'lr' : 1e-5}, {'params': new_params, 'lr': 0.0001}]
    )
    return optimizer


def train_model(data_path,save_model_name, epochs,use_cuda = 0, batch_size = 16):

    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = BertClassifier(device).to(device)
    train_data, val_data, x_train, y_train = get_train_data(data_path)
    optimizer = get_optimizer(model)
    weights = get_loss_weights(y_train)
    
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

#     use_cuda = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss(weight=weights)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                try:
                    output = model(train_input)
                # except Exception as e:
                #     print(e)
                
                    train_label = train_label.to(torch.float)
                    batch_loss = criterion(output, train_label)
                    total_loss_train += batch_loss.item()
                    
                    acc = (torch.argmax(output, dim = 1) == torch.argmax(train_label, dim = 1)).sum().item()
                    total_acc_train += acc

                    model.zero_grad()
                # try:
                    batch_loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(e)
                # break
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in tqdm(val_dataloader):

                    val_label = val_label.to(device)

                    output = model(val_input)
                    val_label = val_label.to(torch.float)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (torch.argmax(output, dim = 1) == torch.argmax(val_label, dim = 1)).sum().item()
                    total_acc_val += acc
            print("saving model - " + save_model_name)
            torch.save(model, save_model_name) 
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f}')# \
#                 | Val Loss: {total_loss_val / len(val_data): .3f} \
#                 | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  