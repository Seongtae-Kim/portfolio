# KUMC BERT PRETRAINING - (VOCAB / WITHOUT-VOCAB)
# Version 1.0.1
# Writer: Seongtae Kim / 2021-03-14

from typing import Type
from torch.utils.data import Dataset
import random
import torch

class InputDataset(Dataset):
    def __init__(self, corpus_path, tokenizer,
                 seq_len=512, encoding="utf-8"):

        from transformers import BertTokenizer
        assert isinstance(tokenizer, BertTokenizer)

        from tqdm import tqdm

        self.seq_len = seq_len
        self.single_seq_len = int(seq_len/2)-1
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            # header 제거
            self.sents = [line[:-1].split("\t") for line in tqdm(f, desc="Dataset Loading...")][1:] # optional
            self.sents_len = len(self.sents)
                    
        print("corpus loaded")

        self.tokenizer = tokenizer
        self.vocab_len = len(tokenizer.vocab)
        self.mask = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        self.cls = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.sep = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.pad = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        # mask token for masked language model
        self.mlmmask = -1
        print("tokenizer loaded")

    def __len__(self):
        return self.sents_len

    def __getitem__(self, item):
        s1, s2, nsp_label = self.random_sent(item)

        s1_rand, s1_label = self.random_word(s1)
        s2_rand, s2_label = self.random_word(s2)

        s1_tokenized = self.tokenizer.tokenize(s1)
        s2_tokenized = self.tokenizer.tokenize(s2)
        

        s1 = [self.cls] + s1_rand[:self.single_seq_len] + [self.sep]
        s2 = s2_rand[:self.single_seq_len+1] + [self.sep] # 하나 더 빼야 마지막 sep 토큰이 안빠짐

        s1_label = [self.mlmmask] + s1_label[:self.single_seq_len] + [self.mlmmask]
        s2_label = s2_label[:self.single_seq_len+1] + [self.sep]


        seg_label = ([0 for _ in range(len(s1))] +
                     [1 for _ in range(len(s2))])[:self.seq_len]
        bert_input = (s1 + s2)[:self.seq_len]
        attention_mask = [1 for _ in range(len(bert_input))][:self.seq_len]
        bert_label = (s1_label + s2_label)[:self.seq_len]

        pads = [self.pad for _ in range(self.seq_len - len(bert_input))]
        mlmpads = [self.mlmmask for _ in range(self.seq_len - len(bert_input))]
        bert_mask= [0]*len(bert_input) + [-1]*(self.seq_len-len(bert_input))

        bert_input.extend(pads)
        bert_label.extend(mlmpads)
        seg_label.extend(pads)
        attention_mask.extend(pads)

        # or 를 and로 수정
        assert (len(bert_input) == len(bert_label)) and (len(bert_input) == len(seg_label)) and (len(bert_input) == len(bert_mask)) and (len(bert_input) == len(attention_mask))

        output = {"bert_input": bert_input,
                  "attention_mask": attention_mask,
                  "bert_mask": bert_mask,
                  "bert_label": bert_label,
                  "seg_label": seg_label,
                  "is_next": nsp_label
                  }
        return {key: torch.tensor(value, dtype=torch.long) for key, value in output.items()}

    def random_word(self, sentence):
        tokenized = self.tokenizer.tokenize(sentence)

        ids = self.tokenizer.convert_tokens_to_ids(tokenized)

        output_label = []

        for i, token in enumerate(ids):
            rand = random.random()

            if rand < 0.15:
                rand /= 0.15
                output_label.append(token)
                
                if rand < 0.8:
                    ids[i] = self.mask
                elif rand < 0.9:
                    ids[i] = random.randrange(self.vocab_len-1)
                else:
                    pass

            else:
                output_label.append(self.mlmmask)

        return ids, output_label

    def random_sent(self, index):
        s1, s2 = self.get_corpus_line(index)

        if random.random() > 0.5:
            return s1, s2, 0
        else:
            return s1, self.get_random_line(), 1

    def get_corpus_line(self, index):
        return self.sents[index][1], self.sents[index][2]

    def get_random_line(self):
        return self.sents[random.randrange(self.sents_len)][2]
