# 2021.02.05, Winter School, Korea University, Seongtae Kim

from typing import overload


class WinterSchool_BERT:
    def __init__(self, train_ratio=None, batch_size=None, epoch=None):
        self.epoch = epoch
        self.batch = batch_size
        self.train_ratio = train_ratio
        self.set_device()
        self.build_BERT()
        self.trained = False

    def set_device(self):
        import torch
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        if not torch.cuda.is_available():
            print(
                "주의! GPU를 사용함으로 설정하지 않으셨습니다. [런타임]-[런타임 유형 설정]에서 GPU 사용으로 설정해주세요.")
        else:
            print("GPU를 사용합니다. {}".format(torch.cuda.get_device_name(0)))

    def get_max_length(self, corpus, verbose=False) -> int:
        mxlen = 0
        for sent in corpus:
            if type(sent) is str:
                input_ids = self.tokenizer.tokenize(sent)
                mxlen = max(mxlen, len(input_ids))
        if verbose:
            print("max length is... ", mxlen)
        return mxlen

    def encode(self, corpus, labels=None, _tqdm=True, verbose=False):
        from tqdm.notebook import tqdm
        import torch

        self.corpus = corpus

        input_ids = []
        attention_masks = []
        if labels is not None:
            assert len(corpus) == len(labels)
        mxlen = self.get_max_length(corpus, verbose)
        if _tqdm:
            for sent in tqdm(corpus):
                self.tokenizer.encode_plus()

                encoded = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=mxlen,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt')
                input_ids.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])
        else:
            for sent in corpus:
                encoded = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=mxlen,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt')
                input_ids.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])

        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_masks = torch.cat(attention_masks, dim=0)

        if labels is not None:
            self.labels = torch.tensor(labels)

    def get_corpus_specifications(self):
        from Korpora import Korpora
        for name, desc in Korpora.corpus_list().items():
            print("{:<40}  {:<}".format(name, desc))

    def build_corpus(self, corpus_name):
        from Korpora import Korpora
        return Korpora.load(corpus_name)

    def build_BERT(self):
        from transformers import BertConfig, BertTokenizer
        self.bert_model_path = "/home/seongtae/SynologyDrive/SIRE/Projects/KR-BERT/KR-BERT/krbert_pytorch/pretrained/pytorch_model_char16424_ranked.bin"
        self.bert_config_path = BertConfig.from_json_file(
            "/home/seongtae/SynologyDrive/SIRE/Projects/KR-BERT/KR-BERT/krbert_pytorch/pretrained/bert_config_char16424.json")
        self.tokenizer = BertTokenizer.from_pretrained(
            '/home/seongtae/SynologyDrive/SIRE/Projects/KR-BERT/KR-BERT/krbert_pytorch/pretrained/vocab_snu_char16424.txt', do_lower_case=False)

    def prepare(self, verbose=False):
        self.build_dataset(verbose)
        self.build_dataloader()
        self.build_optimizer()
        self.build_scheduler()

    def build_scheduler(self):
        from transformers import get_linear_schedule_with_warmup
        self.total_steps = len(self.train_dataloader) * self.epoch
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value in run_glue.py
                                                         num_training_steps=self.total_steps)

    def build_optimizer(self):
        from transformers import AdamW
        self.optimizer = AdamW(self.bert.parameters(), lr=2e-5, eps=1e-8)

    def build_dataset(self, verbose):
        from torch.utils.data import TensorDataset, random_split
        assert self.input_ids != [] and self.attention_masks != []

        if self.labels is not None:
            self.dataset = TensorDataset(
                self.input_ids, self.attention_masks, self.labels)
        else:
            self.dataset = TensorDataset(self.input_ids, self.attention_masks)

        self.train_size = int(self.train_ratio*len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [self.train_size, self.val_size])
        if verbose:
            print('{:>5,} training samples'.format(self.train_size))
            print('{:>5} validation samples'.format(self.val_size))

    def build_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        assert self.train_dataset is not None and self.val_dataset is not None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.batch,
        )
        self.validation_dataloader = DataLoader(
            self.val_dataset,
            sampler=SequentialSampler(self.val_dataset),
            batch_size=self.batch)

    def flat_accuracy(self, preds, labels):
        import numpy as np
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, verbose=True):
        from tqdm.notebook import tqdm
        import random
        import torch
        import numpy as np

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        training_log = []
        desc_training_loss = None
        with tqdm(range(0, self.epoch), leave=False, bar_format="{percentage:2.2f}% {bar} {desc} | {elapsed}>{remaining}") as t:
            for epoch_i in range(0, self.epoch):
                t.update()
                total_train_loss = 0
                self.bert.train()

                for step, batch in enumerate(self.train_dataloader):
                    desc = "epoch: {:,}/{:,} | step: {:,}/{:,}".format(
                        epoch_i+1, len(range(0, self.epoch)), step+1, len(self.train_dataloader))

                    if desc_training_loss is not None:
                        training_log.append(
                            "{:<50}{}".format(desc, desc_training_loss))
                        t.set_description_str(desc+" | "+desc_training_loss)
                    else:
                        t.set_description_str(desc)

                    b_input_ids, b_input_mask, b_labels = map(
                        lambda e: e.to(self.device), batch)

                    self.bert.zero_grad()

                    loss, logits = self.bert(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask,
                                             labels=b_labels)

                    total_train_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.bert.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    avg_train_loss = total_train_loss / \
                        len(self.train_dataloader)
                    desc_training_loss = "mean training loss: {0:.2f}".format(
                        avg_train_loss)
        if verbose:
            for log in training_log:
                print(log)
        self.trained = True

    def validate(self):
        import torch

        self.bert.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        for batch in self.validation_dataloader:
            b_input_ids = batch[0]
            b_input_ids.to(self.device)
            b_input_mask = batch[1]
            b_input_mask.to(self.device)
            b_labels = batch[2]
            b_labels.to(self.device)

            with torch.no_grad():
                (loss, logits) = self.bert(b_input_ids,
                                           token_type_ids=None,
                                           attention_mask=b_input_mask,
                                           labels=b_labels)
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / \
            len(self.validation_dataloader)
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))


class Sentiment_Analysis(WinterSchool_BERT):
    def __init__(self, train_ratio=None, batch_size=None, epoch=None):
        super().__init__(train_ratio, batch_size, epoch)
        self.build()

    def build(self):
        from transformers import BertForSequenceClassification
        self.bert = BertForSequenceClassification.from_pretrained(
            self.bert_model_path, config=self.bert_config_path)
        print("BERT 준비 완료")

    def predict(self, sentence):
        import numpy as np
        import torch

        if not self.trained:
            print("훈련이 먼저 필요합니다. 미훈련된 모델로 테스트합니다.")

        self.get_max_length([sentence])
        self.encode([sentence])

        self.bert.eval()

        with torch.no_grad():
            logit = self.bert(self.input_ids, token_type_ids=None,
                              attention_mask=self.attention_masks)
        return "긍정" if np.argmax(logit[0].detach().cpu().numpy()) else "부정"


class Named_Entity_Recognition(WinterSchool_BERT):
    def __init__(self, train_ratio=None, batch_size=None, epoch=None):
        super().__init__(train_ratio, batch_size, epoch)
        self.build()

    def build(self):
        from transformers import BertForTokenClassification
        self.bert = BertForTokenClassification.from_pretrained(
            self.bert_model_path, config=self.bert_config_path)
        print("BERT 준비 완료")

    @ overload
    def train()
