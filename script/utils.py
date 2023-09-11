import sys

import numpy as np

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, DistilBertTokenizer, ElectraTokenizer,\
                BertModel, DistilBertModel, ElectraModel, \
                DistilBertForSequenceClassification, BertForSequenceClassification, ElectraForSequenceClassification
from model import EnsembleModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# from torch.nn.parallel import DataParallel
import time
import wandb

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

# list all possible 3 mer permutations
def kmer_permutation(list_):
    res = []
    for i in list_:
        for j in list_:
            for k in list_:
                ele = f'{i}{j}{k}'
                res.append(ele)
    return res


# define dataset class
class MyDataset(Dataset):
    def __init__(self, train_data, train_labels, tokenizer_1, tokenizer_2):
        self.data = train_data
        self.labels = train_labels
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        #self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        label = self.labels[index]

        # Tokenize the input text by BERT
        encoding_1 = self.tokenizer_1.encode_plus(
            data_sample,
            add_special_tokens=True,
            truncation=True,
            #padding='max_length',
            #max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids_1 = encoding_1['input_ids'].squeeze()
        attention_mask_1 = encoding_1['attention_mask'].squeeze()

        # Tokenize the input text by DistilBERT
        encoding_2 = self.tokenizer_2.encode_plus(
            data_sample,
            add_special_tokens=True,
            truncation=True,
            #padding='max_length',
            #max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids_2 = encoding_2['input_ids'].squeeze()
        attention_mask_2 = encoding_2['attention_mask'].squeeze()

        # Return the input ids, attention mask, and label as tensors
        return input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, torch.tensor(label)








