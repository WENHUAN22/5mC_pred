import numpy as np
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, ElectraTokenizer,\
                        DistilBertModel, ElectraModel, \
                        DistilBertForSequenceClassification, ElectraForSequenceClassification
from model import EnsembleModel, AttentionCheck, EnsembleModel_baseline
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef

import time
from utils import seq2kmer, kmer_permutation, MyDataset

# load training data
mydf = pd.read_csv('../data/train_dataset.csv', sep='\t')
mydf.loc[mydf.label=='non-Methyl', 'label'] = 0
mydf.loc[mydf.label=='Methyl', 'label'] = 1

mydf['text'] = list(map(lambda x: seq2kmer(x, 3), mydf['seq']))

list_3mer = kmer_permutation(['A','T','G','C'])

# prepare dataset for training and evaluation
train_set, valid_set = train_test_split(mydf, test_size=0.2, stratify=mydf['label'], random_state=100)
train_set.reset_index(drop=True, inplace=True)
valid_set.reset_index(drop=True, inplace=True)
train_data, train_labels = train_set['text'], train_set['label']
valid_data, valid_labels = valid_set['text'], valid_set['label']

# Define the BERT model name and tokenizer
tokenizerDilstilBERT = DistilBertTokenizer.from_pretrained('wenhuan/MuLan-Methyl-DistilBERT')
tokenizerELECTRA = ElectraTokenizer.from_pretrained('wenhuan/MuLan-Methyl-ELECTRA')
# model
model_distilbert = DistilBertModel.from_pretrained(f'wenhuan/MuLan-Methyl-DistilBERT_5hmC', num_labels=2, output_attentions=True)
model_electra = ElectraModel.from_pretrained(f'wenhuan/MuLan-Methyl-ELECTRA_5hmC', num_labels=2, output_attentions=True)
attnCheck = AttentionCheck()

model = EnsembleModel(model_distilbert, model_electra, attnCheck, num_classes=2)
#model = EnsembleModel_baseline(model_distilbert, model_electra, num_classes=2)

model_dict = {'DilstilBERT': [tokenizerDilstilBERT, model_distilbert],
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


# Create an instance of the TrainDataset
train_dataset = MyDataset(train_data, train_labels, tokenizerDilstilBERT, tokenizerELECTRA)
valid_dataset = MyDataset(valid_data, valid_labels, tokenizerDilstilBERT, tokenizerELECTRA)


# Define the hyperparameters
batch_size = 256
#learning_rate = 2e-4
learning_rate = 2e-5
num_epochs = 32

# Create the train DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seed for reproducibility
torch.manual_seed(100)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#early stop
class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Train the ensemble model
print(f'start model training')
best_loss = 0.0
best_auc = 0.0
best_loss = float('inf')

early_stopper = EarlyStopper(patience=6, min_delta=0)


model.to(device)
model = nn.DataParallel(model)

for epoch in range(num_epochs):
    start_time = time.time()
    # Create an instance of the ensemble model
    model.train()
    model.to(device)
    # model = DataParallel(model, device_ids=[0,1,2,3])

    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for batch in train_loader:

        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch
        input_ids_1, input_ids_2, labels = input_ids_1.to(device), input_ids_2.to(device), labels.to(device)

        if input_ids_1.shape[1] == input_ids_2.shape[1] == 41:
            pass
        else:
            print('tokenizer size error')

        # zero the gradients
        optimizer.zero_grad()
        # Forward pass
        logits = model(input_ids_1, input_ids_2)
        # Compute the loss
        loss = criterion(logits, labels)
        # Backward pass
        loss.backward()
        # update parameters
        optimizer.step()
        # Record the training loss
        running_loss += loss.item()
        # Record the training accuracy
        probability = nn.functional.softmax(logits, dim=1)
        #probability = logits
        predicted = torch.argmax(probability, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # AUC
        y_true.extend(labels.tolist())
        y_pred.extend(probability[:,1].tolist())

    train_loss = running_loss/len(train_loader)
    train_acc = correct / total
    train_auc = roc_auc_score(y_true, y_pred)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {train_loss:.4f} Accuracy: {train_acc: .4f} AUC: {train_auc: .4f} Training Time: {epoch_time} seconds")

    # Evaluate the ensemble model

    model.eval()
    model.to(device)

    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    valid_y_true = []
    valid_y_prob = []
    valid_y_pred = []

    with torch.no_grad():
        correct = 0
        total = 0

        for _batch in valid_loader:

            _input_ids_1, _attention_mask_1, _input_ids_2, _attention_mask_2, _labels = _batch
            _input_ids_1, _input_ids_2, _labels = _input_ids_1.to(device), _input_ids_2.to(device), _labels.to(device)

            _logits = model(_input_ids_1, _input_ids_2)
            _loss = criterion(_logits, _labels)

            valid_loss += _loss.item() * _input_ids_1.size(0)
            _probability = nn.functional.softmax(_logits, dim=1)
            #_probability = _logits
            _predicted = torch.argmax(_probability, dim=1)
            valid_correct += (_predicted == _labels).sum().item()
            valid_total += _labels.size(0)

            valid_y_true.extend(_labels.tolist())
            valid_y_prob.extend(_probability[:,1].tolist())
            valid_y_pred.extend(_predicted.tolist())


        _accuracy = valid_correct / valid_total
        _auc = roc_auc_score(valid_y_true, valid_y_prob)
        _loss = valid_loss / valid_total
        _f1 = f1_score(valid_y_true, valid_y_pred)
        _recall = recall_score(valid_y_true, valid_y_pred)
        _precision = recall_score(valid_y_true, valid_y_pred)
        _mcc = matthews_corrcoef(valid_y_true, valid_y_pred)

        print(f'Epoch {epoch+1}: Validation Loss: {_loss: .4f}, Accuracy: {_accuracy: .4f} AUC: {_auc: .4f} F1-score: {_f1: .4f} Recall: {_recall: .4f} Precision: {_precision: .4f} MCC: {_mcc: .4f}')


        # save the best model
        if _loss < best_loss:
            best_loss = _loss
            torch.save(model.module.state_dict(), f'../model/EA_5mC_{epoch+1}_{_loss:.4f}.pth')

    # early stop
    if early_stopper.early_stop(_loss):
        print(f'early stopped at epoch {epoch+1}')
        break

