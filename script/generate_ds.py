'''
this program is producing training dataset for explainable ai project
'''
import pandas as pd
from ast import literal_eval
import numpy as np
import re
from collections import Counter

# read training database and test database
list_ = ['train', 'test']
data_type = list_[1]
annotated_df = pd.read_csv(f'/mnt/volume/project/x5mC/db/initial/{data_type}_raw_annotated_data_8hc.txt', sep='\t',converters={'pos_index': literal_eval, 'ref_genotype': literal_eval, 'project': literal_eval})
annotated_df['pos_index_0'] = list(map(lambda x,y: (np.array(x)-y).tolist(), annotated_df['pos_index'], annotated_df['chromStart']))
annotated_df['pos_index_0'] = list(map(lambda x: list(set(x)), annotated_df['pos_index_0']))
# extract 41 bp length sequence to generate training dataset
# positive samples index
def func_extractSeq(longSeq, index, genoType=None):
    seq_list = []
    for i in range(len(index)):
        if index[i] >= 20 and index[i] <= 978:
            ele = index[i]
            seq = longSeq[ele-20:ele+21]
            if genoType != None:
                seq_list.append((seq, genoType[i]))
            else:
                seq_list.append(seq)
    return seq_list

df_1 = annotated_df
df_1['shortSeq'] = list(map(lambda x,y,z: func_extractSeq(x, y, z), df_1['gene_promoter_region'], df_1['pos_index_0'], df_1['ref_genotype']))

seqList_pos = list(df_1['shortSeq'])
seqList_pos = sum(seqList_pos, [])

pos_df = pd.DataFrame(columns=['seq', 'genotype', 'label'])
pos_df['seq'], pos_df['genotype'], pos_df['label'] = list(map(lambda x: x[0], seqList_pos)), list(map(lambda x: x[1], seqList_pos)), 'Methyl'
pos_df.drop_duplicates(keep='first', inplace=True, subset='seq') # 91296

# negative samples index
def func_negIdx(longSeq, posIndex):
    c_index = [substr.start() for substr in re.finditer('C', longSeq)]
    negIndex = list(set(c_index)-set(posIndex))
    return negIndex

df_2 = annotated_df
df_2['neg_index_0'] = list(map(lambda x,y: func_negIdx(x, y), df_2['gene_promoter_region'], df_2['pos_index_0']))
df_2['shortSeq'] = list(map(lambda x,y: func_extractSeq(x, y, None), df_2['gene_promoter_region'], df_2['neg_index_0']))

seqList_neg = list(df_2['shortSeq'])
seqList_neg = sum(seqList_neg, [])

neg_df = pd.DataFrame(columns=['seq', 'label'])
neg_df['seq'] = seqList_neg
neg_df['label'] = 'non-Methyl'
neg_df.drop_duplicates(keep='first', inplace=True, subset='seq')
print(f'number of neg_df {len(neg_df)}')
neg_df.to_csv(f'/mnt/volume/project/x5mC/db/{data_type}_dataset_neg.csv', sep='\t', index=False)

tmp_all = pos_df[['seq', 'label']].append(neg_df)
tmp_all.reset_index(drop=True, inplace=True)
tmp_all.drop_duplicates(keep='first', inplace=True, subset='seq')
# training dataset
'''
tmp_neg_df = tmp_all[tmp_all['label'] == 'non-Methyl']
tmp_neg_df = tmp_neg_df.sample(n=len(pos_df), random_state=22) # training dataset
print(f'number of tmp_neg_df {len(tmp_neg_df)}')

all_df = pos_df[['seq', 'label']].append(tmp_neg_df)
all_df.drop_duplicates(keep='first', subset='seq', inplace=True)
'''
all_df = tmp_all
all_df = all_df.sample(frac=1)
all_df.reset_index(drop=True, inplace=True)
print(all_df)

all_df.to_csv(f'/mnt/volume/project/x5mC/db/{data_type}_dataset_new.csv', sep='\t', index=False)


#train_data = pd.read_csv('/mnt/volume/project/x5mC/db/train_dataset_new.csv', sep='\t') # 182718 rows
#Counter(train_data['label']) # 91296 for each
#tmp = train_data.drop_duplicates(subset='seq', keep='first') # 182581 rows

#Counter(tmp['label'])


# generate test dataset



