import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

class EnsembleModel_baseline(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super(EnsembleModel_baseline, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.resize_linear = nn.Linear(768, 256)
        self.classification_linear = nn.Linear(256, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(256)



    def forward(self, x1, x2):
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        output1_hidden_state = self.resize_linear(output1.last_hidden_state[:,0,:]) # [batch_size, 768] vector of CLS
        output2_hidden_state = output2.last_hidden_state[:,0,:]
        stacked_hidden_state = torch.stack((output1_hidden_state, output2_hidden_state), dim=1)
        stacked_hidden_state_avg = torch.mean(stacked_hidden_state, dim=1) #[batch_size, 256]

        bn_outputs = self.bn(stacked_hidden_state_avg)
        act_outputs = self.activation(bn_outputs)
        outputs = self.dropout(act_outputs)
        logits = self.classification_linear(outputs)

        return logits


class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, attnCheck, num_classes):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.attnCheck = attnCheck
        self.resize_linear = nn.Linear(768, 256)
        self.classification_linear = nn.Linear(256+36, num_classes)
        self.sub_dropout = nn.Dropout(p=0.2)
        self.sub_activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(256)
        self.normal = nn.LayerNorm(256+36)

    def forward(self, x1, x2):
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        output1_hidden_state = self.resize_linear(output1.last_hidden_state[:,0,:])
        output1_hidden_state = self.bn(output1_hidden_state)
        output1_hidden_state = self.sub_activation(output1_hidden_state)
        output1_hidden_state = self.sub_dropout(output1_hidden_state)

        output2_hidden_state = output2.last_hidden_state[:,0,:]
        output2_hidden_state = self.bn(output2_hidden_state)
        output2_hidden_state = self.sub_activation(output2_hidden_state)
        output2_hidden_state = self.sub_dropout(output2_hidden_state)

        stacked_hidden_state = torch.stack((output1_hidden_state, output2_hidden_state), dim=1)
        stacked_hidden_state_avg = torch.mean(stacked_hidden_state, dim=1) #[batch_size, 256]

        stacked_hidden_state_avg = self.sub_activation(stacked_hidden_state_avg)
        stacked_hidden_state_avg = self.sub_dropout(stacked_hidden_state_avg)

        # obtain the 19th, 20th, 21st tokens attention scores' rank among all layer
        for i in range(12):
            if i < 6:
                attn1 = output1.attentions[i]
                attn2 = output2.attentions[i]
            else:
                attn1 = None
                attn2 = output2.attentions[i]
            if i == 0:
                layer_attn_rank = self.attnCheck(attn1, attn2)
            else:
                layer_attn_rank_ = self.attnCheck(attn1, attn2)
                layer_attn_rank = torch.cat((layer_attn_rank, layer_attn_rank_), dim=1)

        combine_output = torch.cat((stacked_hidden_state_avg, torch.pow(layer_attn_rank, -1)), dim=1)
        combine_output = self.normal(combine_output)
        activate_output = self.sub_activation(combine_output)
        activate_output = self.sub_dropout(activate_output)
        logits = self.classification_linear(activate_output)

        return logits

class AttentionCheck(nn.Module):
    def __init__(self):
        super(AttentionCheck, self).__init__()

    def rank(self, x):
        # input is the layer-wise attention matrix [batch_size, head, token, token]
        cls_attn_avg_head = torch.mean(x[:, :, 0, :], dim=1) # [batch_size, 41]
        _, idx = cls_attn_avg_head.sort(dim=1, descending=False)
        # find the position of 19,20,21
        rank_19 = (idx==19).nonzero(as_tuple=False)
        rank_20 = (idx==20).nonzero(as_tuple=False)
        rank_21 = (idx==21).nonzero(as_tuple=False)
        rank_allSite = torch.stack((rank_19[:,1], rank_20[:,1], rank_21[:,1]), dim=1) + 1 # [batch_size, 3]
        return rank_allSite

    def forward(self, attn1, attn2):
        if attn1 != None:
            rank_allSite_model1 = self.rank(attn1)
            rank_allSite_model2 = self.rank(attn2)
            models_stack = torch.stack((rank_allSite_model1.float(), rank_allSite_model2.float()), dim=1)
        else:
            rank_allSite_model2 = self.rank(attn2)
        if attn1 != None:
            rank_allSite_models_avg = torch.mean(models_stack, dim=1) # [batch_size, 3]
        else:
            rank_allSite_models_avg = rank_allSite_model2
        return rank_allSite_models_avg