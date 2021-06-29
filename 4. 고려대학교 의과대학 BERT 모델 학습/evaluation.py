# KUMC BERT PRETRAINING - (VOCAB / WITHOUT-VOCAB)
# Version 1.0.1
# Writer: Seongtae Kim / 2021-03-10
# Revision: Yoojoong Kim, Ph.D. / 2021-04-02

from pretrain.examples.extract_features import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from kumc_dataset import InputDataset
from pynvml.smi import nvidia_smi
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
import numpy as np
import pickle, re
import torch
import json
import os

def to_np(t):
    return t.cpu().detach().numpy()



# HYPERPARAMETERS
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
batch_size = 32
seq_len = 128
epoch=20
n_gpu = torch.cuda.device_count()
with_vocab=False
gpu_write=False
savepath="./models/"

tokenizer = BertTokenizer(vocab_file="./kr_bert_vocab.txt", do_lower_case=False)    # KUMC BERT TOKENIZER (without vocab)
model = BertForPreTraining.from_pretrained("./kr_bert_model/")                      # KUMC BERT (without vocab)
# model = BertForPreTraining.from_pretrained("./models/without_vocab/kumc_bert(TRAIN_e5).mdl")
savepath+="without_vocab/"

# model = torch.load("./models/without_vocab_old/kumc_bert(TRAIN_e5).mdl", map_location='cpu')

loss_fct = CrossEntropyLoss(ignore_index=-1).to(device)

# DATASET & DATALOADER
print("TRAINING DATASET")
# dataset = InputDataset("./whole_train.tsv", tokenizer=tokenizer, seq_len=seq_len)
print()
print("VALIDATION DATASET")
val_dataset = InputDataset("./whole_test.tsv", tokenizer=tokenizer, seq_len=seq_len)
print()

# dataloader = DataLoader(dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

e=0

str_code = "VALIDATION"

loss_sum=0.0
mask_loss=0.0
nsp_loss=0.0
mlm_acc=0.0
nsp_acc=0.0
data_iter = tqdm(enumerate(val_dataloader),
                    desc="Epoch_%s:%d" % (str_code, e+1),
                    total=len(val_dataloader),
                    bar_format="{l_bar}{r_bar}")



# Validation loop
for i, data in data_iter:
    # torch.cuda.empty_cache()
    data = {key: value.to(device) for key, value in data.items()}
    sequence_output, pooled_output = model.bert(input_ids=data["bert_input"],
                                                       token_type_ids=data["seg_label"],
                                                       attention_mask=data["attention_mask"],
                                                       output_all_encoded_layers=False)
    prediction_scores, seq_relationship_score = model.cls(sequence_output, pooled_output)
    masked_lm_labels, next_sentence_label= data["bert_label"], data["is_next"]
    masked_lm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), masked_lm_labels.view(-1))
    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
    _, masked_lm_pred = torch.max(prediction_scores,2)
    _, next_sentence_pred = torch.max(seq_relationship_score, 1)
    masked_lm_acc = (masked_lm_labels==masked_lm_pred)[data['bert_label']>=0].float().sum() / (data['bert_label']>=0).sum()
    next_sentence_acc = (next_sentence_label==next_sentence_pred).float().sum() / next_sentence_pred.shape[0]
    total_loss = masked_lm_loss + next_sentence_loss
    loss_sum+=to_np(total_loss.sum())
    mask_loss+=to_np(masked_lm_loss)
    nsp_loss+=to_np(next_sentence_loss)
    mlm_acc+=to_np(masked_lm_acc)
    nsp_acc+=to_np(next_sentence_acc)
    print("Loss [MLM/NSP]: [%.4f / %.4f]\nAccuracy [MLM/NSP]: [%.4f / %.4f]"%(mask_loss/(i+1), nsp_loss/(i+1), mlm_acc/(i+1), nsp_acc/(i+1)))
    if i==10:
        with open('/models/kr_bert/validation.tsv', 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['MLM loss', 'NSP loss', 'MLM Acc.', 'NSP Acc.'])
            tsv_writer.writerow([mask_loss/(i+1), nsp_loss/(i+1),
                                 mlm_acc/(i+1), nsp_acc/(i+1)])
        break


