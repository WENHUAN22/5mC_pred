import numpy as np
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, ElectraTokenizer, \
    DistilBertModel, ElectraModel, \
    DistilBertForSequenceClassification, ElectraForSequenceClassification
from model import EnsembleModel, AttentionCheck, EnsembleModel_baseline
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix

import time
from utils import seq2kmer, kmer_permutation, MyDataset



# load data
mydf = pd.read_csv(f'../data/test_dataset.csv', sep='\t') #2098913
mydf.loc[mydf.label=='non-Methyl', 'label'] = 0 # 2082786
mydf.loc[mydf.label=='Methyl', 'label'] = 1 # 16127

# subsample negative sample
mydf_pos = mydf[mydf['label'] == 1]
mydf_neg = mydf[mydf['label'] == 0]
sub_mydf_neg = mydf_neg.sample(n=10*len(mydf_pos), random_state=22)

new_mydf = pd.concat([mydf_pos, sub_mydf_neg])
mydf = new_mydf
mydf.reset_index(inplace=True, drop=True)
print(Counter(mydf['label']))

mydf['text'] = list(map(lambda x: seq2kmer(x, 3), mydf['seq']))

# list all possible 3 mer permutations
list_3mer = kmer_permutation(['A','T','G','C'])

tokenizerDilstilBERT = DistilBertTokenizer.from_pretrained('wenhuan/MuLan-Methyl-DistilBERT')
tokenizerELECTRA = ElectraTokenizer.from_pretrained('wenhuan/MuLan-Methyl-ELECTRA')

# model
model_distilbert = DistilBertModel.from_pretrained('wenhuan/MuLan-Methyl-DistilBERT_5hmC', num_labels=2, output_attentions=True)
model_electra = ElectraModel.from_pretrained('wenhuan/MuLan-Methyl-ELECTRA_5hmC', num_labels=2, output_attentions=True)
attnCheck = AttentionCheck()
model = EnsembleModel(model_distilbert, model_electra, attnCheck, num_classes=2)
#model = EnsembleModel_baseline(model_distilbert, model_electra, num_classes=2)



model_dict = {
              'DilstilBERT': [tokenizerDilstilBERT, model_distilbert],
              'ELECTRA': [tokenizerELECTRA, model_electra]}

# check if all 3mer is included in the vocabulary

for ele_ in ['DilstilBERT', 'ELECTRA']:
    tokenizer_, model_ = model_dict[ele_][0], model_dict[ele_][1]
    if ele_ == 'ELECTRA':
        list_3mer = [x.lower() for x in list_3mer]
    # new tokens
    new_tokens = set(list_3mer) - set(tokenizer_.vocab.keys())
    # add tokens to the tokenizer vocabulary
    tokenizer_.add_tokens(list(new_tokens))
    print(len(tokenizer_))
    # add new, random embeddings for the new tokens
    model_.resize_token_embeddings(len(tokenizer_))

test_data = mydf['text']
test_label = mydf['label']

test_dataset = MyDataset(test_data, test_label, tokenizerDilstilBERT, tokenizerELECTRA)

# Define the hyperparameters
batch_size = 256
# Create the DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seed for reproducibility
torch.manual_seed(100)


model.load_state_dict(torch.load(f'./model/EA_5mC_model.pth'))

model.eval()
model.to(device)
pred_labels = []
pred_probs = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch
        input_ids_1, input_ids_2, labels = input_ids_1.to(device), input_ids_2.to(device), labels.to(device)

        logits = model(input_ids_1, input_ids_2)
        probs = nn.functional.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1)
        pred_labels.extend(predicted.tolist())
        pred_probs.extend(probs[:,1].tolist())
        true_labels.extend(labels.tolist())

    acc = accuracy_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_probs)
    f1 = f1_score(true_labels, pred_labels)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)

    print(f'accuracy {acc: .4f} auc {auc: .4f} f1 {f1: .4f} mcc {mcc: .4f}')
    print(cm)
