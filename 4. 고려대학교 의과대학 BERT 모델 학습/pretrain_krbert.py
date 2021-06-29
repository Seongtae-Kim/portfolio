# KUMC BERT PRETRAINING - (VOCAB / WITHOUT-VOCAB)
# Version 1.0.1
# Writer: Seongtae Kim / 2021-03-10

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
import csv

class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def to_np(t):
    return t.cpu().detach().numpy()

torch.cuda.empty_cache()

# HYPERPARAMETERS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
seq_len = 128
epoch=100
n_gpu = torch.cuda.device_count()
with_vocab=False
gpu_write=False
savepath="./models/"

# MODELS & TOKENIZERS (KR-BERT BASELINE / KUMC BERT)
if with_vocab:
    tokenizer = BertTokenizer(vocab_file="./kumc_vocab_bert.txt", do_lower_case=False)   # KUMC BERT TOKENIZER
    model = BertForPreTraining.from_pretrained("./model_ranked/")                        # KUMC BERT
    savepath+="with_vocab/"
else:
    tokenizer = BertTokenizer(vocab_file="./kr_bert_vocab.txt", do_lower_case=False)    # KUMC BERT TOKENIZER (without vocab)
    model = BertForPreTraining.from_pretrained("./kr_bert_model/")                      # KUMC BERT (without vocab)
    savepath+="without_vocab/"

# Loading previous best info - DEPRECATED
if "kumc_bert.log" in os.listdir(savepath):
    log = open(savepath+"kumc_bert.log").read()
    prev_best=float(re.findall("(?!'total loss avg': )\d+\.\d+", log)[0])
else:
    prev_best=999.999


model.to(device)
loss_fct = CrossEntropyLoss(ignore_index=-1).to(device)

# DATASET & DATALOADER
print("TRAINING DATASET")
dataset = InputDataset("./whole_train.tsv", tokenizer=tokenizer, seq_len=seq_len)
print()
print("VALIDATION DATASET")
val_dataset = InputDataset("./whole_test.tsv", tokenizer=tokenizer, seq_len=seq_len)
print()

dataloader = DataLoader(dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

print()
print("")
print("\n==KUMC BERT Pretrainer==\nWriter: Seongtae Kim\nWritten Date:2021-03-16\nSave path: {}\nTotal epochs: {}\nDevice: {}\nGPUs: {}\nPrev best:{}\nBatch size:{}\nSeq len:{}\n".format(savepath, epoch, device, n_gpu, prev_best, batch_size, seq_len))
print()

optim = Adam(model.parameters(), lr=float(1e-4), betas=(0.9, 0.999), weight_decay=0.01)
optim_schedule = ScheduledOptim(optim, 768, n_warmup_steps=10000)
log_freq=1000



# ITERATION
for e in range(epoch):
    loss_sum=0.0
    mask_loss=0.0
    nsp_loss=0.0
    mlm_acc = 0.0
    nsp_acc = 0.0
    str_code = "TRAIN"

    data_iter = tqdm(enumerate(dataloader),
                        desc="Epoch_%s:%d" % (str_code, e+1),
                        total=len(dataloader),
                        bar_format="{l_bar}{r_bar}")

    with open(savepath+'log/' + ('train_%03d.tsv'%(e+1)), 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['Iter.', 'MLM loss', 'NSP loss', 'MLM Acc.', 'NSP Acc.'])
        # Training loop
        for i, data in data_iter:
            torch.cuda.empty_cache()
            data = {key: value.to(device) for key, value in data.items()}

            sequence_output, pooled_output = model.bert(input_ids=data["bert_input"],
                                                               token_type_ids=data["seg_label"],
                                                               attention_mask=data["attention_mask"],
                                                               output_all_encoded_layers=False)

            prediction_scores, seq_relationship_score = model.cls(sequence_output, pooled_output)

            masked_lm_labels, next_sentence_label= data["bert_label"], data["is_next"]

            # CrossEntropy Loss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

            # Accuracy
            _, masked_lm_pred = torch.max(prediction_scores, 2)
            _, next_sentence_pred = torch.max(seq_relationship_score, 1)
            masked_lm_acc = (masked_lm_labels==masked_lm_pred)[data['bert_label'] >= 0].float().sum() / (
                        data['bert_label'] >= 0).sum()
            next_sentence_acc = (next_sentence_label==next_sentence_pred).float().sum() / next_sentence_pred.shape[0]

            if str_code == "TRAIN":
                optim_schedule.zero_grad()
                total_loss.sum().backward()
                optim_schedule.step_and_update_lr()

            loss_sum+=to_np(total_loss.sum())
            mask_loss+=to_np(masked_lm_loss)
            nsp_loss+=to_np(next_sentence_loss)
            mlm_acc += to_np(masked_lm_acc)
            nsp_acc += to_np(next_sentence_acc)

            post_fix = {
                "epoch":e+1,
                "iter":i+1,
                "loss": round(float(to_np(total_loss.sum())), 4),
                "loss avg": round(loss_sum / (i+1), 4),
                "mask loss": round(mask_loss.sum() / (i+1), 4),
                "nsp loss": round(nsp_loss.sum() / (i+1), 4),
                "mlm acc": round(mlm_acc.sum()/ (i+1), 4),
                "nsp acc": round(nsp_acc.sum()/ (i+1), 4)
            }
            if i % log_freq == 0:
                tsv_writer.writerow([i + 1,
                                     round(to_np(masked_lm_loss), 4),
                                     round(to_np(masked_lm_acc), 4),
                                     round(to_np(masked_lm_acc), 4),
                                     round(to_np(next_sentence_acc), 4)])
                gpu_stats=[]
                nvsmi = nvidia_smi.getInstance()
                gpus = nvsmi.DeviceQuery("memory.free, memory.total")["gpu"]

                for i, gpu in enumerate(gpus):
                    total = round(gpu["fb_memory_usage"]["total"])
                    free = round(gpu["fb_memory_usage"]["free"])
                    used = round(total - free)
                    used_perc = round(used /total * 100, 2)

                    if i == len(gpus)-1:
                        gpu_stats.append("GPU {}: {}%".format(i, used_perc))
                    else:
                        gpu_stats.append("GPU {}: {}% | ".format(i, used_perc))

                if gpu_write:
                    data_iter.write(str(post_fix)+"\n"+"".join(gpu_stats))
                else:
                    data_iter.write(str(post_fix))

            if loss_sum / (i+1) < prev_best:
                prev_best = loss_sum / (i+1)
                name = "kumc_bert({}_e{})".format(str_code, e+1)
                torch.save(model, savepath+name+".mdl")
                json.dump(post_fix, open(savepath+name+".log", "w"))
                open(savepath+name+".log", "w").write(str(post_fix))
                #data_iter.write("MODEL SAVED ({}: epoch={})".format(str_code, e+1))

            del data
            torch.cuda.empty_cache()

        else:
            tsv_writer.writerow([i + 1,
                                 round(mask_loss / (i + 1), 4),
                                 round(nsp_loss / (i + 1), 4),
                                 round(mlm_acc / (i + 1), 4),
                                 round(nsp_acc / (i + 1), 4)])

    # VALIDATION
    str_code = "VALIDATION"

    loss_sum=0.0
    mask_loss=0.0
    nsp_loss=0.0
    mlm_acc = 0.0
    nsp_acc = 0.0
    data_iter = tqdm(enumerate(val_dataloader),
                        desc="Epoch_%s:%d" % (str_code, e+1),
                        total=len(val_dataloader),
                        bar_format="{l_bar}{r_bar}")

    with open(savepath+'log/' + ('validation_%03d.tsv'%(e+1)), 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['Iter.', 'MLM loss', 'NSP loss', 'MLM Acc.', 'NSP Acc.'])
        # Validation loop
        for i, data in data_iter:
            torch.cuda.empty_cache()
            data = {key: value.to(device) for key, value in data.items()}

            sequence_output, pooled_output = model.bert(input_ids=data["bert_input"],
                                                               token_type_ids=data["seg_label"],
                                                               attention_mask=data["attention_mask"],
                                                               output_all_encoded_layers=False)

            prediction_scores, seq_relationship_score = model.cls(sequence_output, pooled_output)

            masked_lm_labels, next_sentence_label= data["bert_label"], data["is_next"]

            # CrossEntropy Loss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

            # Accuracy
            _, masked_lm_pred = torch.max(prediction_scores, 2)
            _, next_sentence_pred = torch.max(seq_relationship_score, 1)
            masked_lm_acc = (masked_lm_labels==masked_lm_pred)[data['bert_label'] >= 0].float().sum() / (
                        data['bert_label'] >= 0).sum()
            next_sentence_acc = (next_sentence_label==next_sentence_pred).float().sum() / next_sentence_pred.shape[0]

            loss_sum+=to_np(total_loss.sum())
            mask_loss+=to_np(masked_lm_loss)
            nsp_loss+=to_np(next_sentence_loss)
            mlm_acc += to_np(masked_lm_acc)
            nsp_acc += to_np(next_sentence_acc)

            post_fix = {
                "epoch":e+1,
                "iter":i+1,
                "loss": round(float(to_np(total_loss.sum())), 4),
                "loss avg": round(loss_sum / (i+1), 4),
                "mask loss": round(mask_loss.sum() / (i+1), 4),
                "nsp loss": round(nsp_loss.sum() / (i+1), 4),
                "mlm acc": round(mlm_acc.sum() / (i + 1), 4),
                "nsp acc": round(nsp_acc.sum() / (i + 1), 4)
            }

            if i % log_freq == 0:
                tsv_writer.writerow([i + 1,
                                     round(to_np(masked_lm_loss), 4),
                                     round(to_np(next_sentence_loss), 4),
                                     round(to_np(masked_lm_acc), 4),
                                     round(to_np(next_sentence_acc), 4)])
                gpu_stats=[]
                nvsmi = nvidia_smi.getInstance()
                gpus = nvsmi.DeviceQuery("memory.free, memory.total")["gpu"]

                for i, gpu in enumerate(gpus):
                    total = round(gpu["fb_memory_usage"]["total"])
                    free = round(gpu["fb_memory_usage"]["free"])
                    used = round(total - free)
                    used_perc = round(used /total * 100, 2)

                    if i == len(gpus)-1:
                        gpu_stats.append("GPU {}: {}%".format(i, used_perc))
                    else:
                        gpu_stats.append("GPU {}: {}% | ".format(i, used_perc))

                if gpu_write:
                    data_iter.write(str(post_fix)+"\n"+"".join(gpu_stats))
                else:
                    data_iter.write(str(post_fix))

            del data
            torch.cuda.empty_cache()

        else:
            tsv_writer.writerow([i + 1,
                                 round(mask_loss / (i + 1), 4),
                                 round(nsp_loss / (i + 1), 4),
                                 round(mlm_acc / (i + 1), 4),
                                 round(nsp_acc / (i + 1), 4)])

    # MODEL SAVING
    prev_best = loss_sum / (i+1)
    name = "kumc_bert({}_e{})".format(str_code, e+1)
    torch.save(model, savepath+name+".mdl")
    json.dump(post_fix, open(savepath+name+".log", "w"))
    open(savepath+name+".log", "w").write(str(post_fix))
    data_iter.write("MODEL SAVED ({}: epoch={})".format(str_code, e+1))

print("FINISHED")