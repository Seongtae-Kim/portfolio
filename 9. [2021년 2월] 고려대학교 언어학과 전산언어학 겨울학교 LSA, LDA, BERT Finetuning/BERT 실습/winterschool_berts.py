class Models:
    def __init__(self) -> None:
        self.lists = {}

        # M-BERT
        from transformers import BertTokenizerFast, BertForMaskedLM
        self.bert_multilingual_tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-multilingual-cased')
        self.bert_multilingual_model = BertForMaskedLM.from_pretrained(
            'bert-base-multilingual-cased').eval()
        self.lists["M-BERT"] = {"Tokenizer": self.bert_multilingual_tokenizer,
                                "Model": self.bert_multilingual_model}
        print("Google Multilingual BERT 로드 완료")

        # KR-BERT
        from transformers import BertTokenizerFast, BertForMaskedLM
        self.krbert_tokenizer = BertTokenizerFast.from_pretrained(
            'snunlp/KR-Medium')
        self.krbert_model = BertForMaskedLM.from_pretrained(
            'snunlp/KR-Medium').eval()
        self.lists["KR-Medium"] = {"Tokenizer": self.krbert_tokenizer,
                                   "Model": self.krbert_model}
        print("KR-BERT 로드 완료")

        # BERT
        from transformers import BertTokenizerFast, BertForMaskedLM
        self.bert_kor_tokenizer = BertTokenizerFast.from_pretrained(
            'kykim/bert-kor-base')
        self.bert_kor_model = BertForMaskedLM.from_pretrained(
            'kykim/bert-kor-base').eval()
        self.lists["bert-kor-base"] = {"Tokenizer": self.bert_kor_tokenizer,
                                       "Model": self.bert_kor_model}
        print("BERT-kor-base 로드 완료")

        # ALBERT
        from transformers import AlbertForMaskedLM
        self.albert_tokenizer = BertTokenizerFast.from_pretrained(
            'kykim/albert-kor-base')
        self.albert_model = AlbertForMaskedLM.from_pretrained(
            'kykim/albert-kor-base').eval()
        self.lists["albert-kor-base"] = {"Tokenizer": self.albert_tokenizer,
                                         "Model": self.albert_model}
        print("ALBERT-kor-base 로드 완료")

        # XLM-Roberta
        from transformers import XLMRobertaTokenizerFast, XLMRobertaForMaskedLM
        self.xlmroberta_tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            'xlm-roberta-base')
        self.xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained(
            'xlm-roberta-base').eval()
        self.lists["xlm-roberta-base"] = {"Tokenizer": self.xlmroberta_tokenizer,
                                          "Model": self.xlmroberta_model}
        print("XLM-Roberta-kor 로드 완료")

        # electra-base-kor
        from transformers import ElectraTokenizerFast, ElectraModel
        self.tokenizer_electra = ElectraTokenizerFast.from_pretrained(
            "kykim/electra-kor-base")
        self.electra_model = ElectraModel.from_pretrained(
            "kykim/electra-kor-base")
        self.lists["electra-kor-base"] = {"Tokenizer": self.tokenizer_electra,
                                          "Model": self.electra_model}
        print("electra-kor-base 로드 완료")

        # gpt3-kor-small_based_on_gpt2
        from transformers import BertTokenizerFast, GPT2LMHeadModel
        self.tokenizer_gpt3 = BertTokenizerFast.from_pretrained(
            "kykim/gpt3-kor-small_based_on_gpt2")
        self.model_gpt3 = GPT2LMHeadModel.from_pretrained(
            "kykim/gpt3-kor-small_based_on_gpt2")
        self.lists["gpt3-kor-small_based_on_gpt2"] = {"Tokenizer": self.tokenizer_gpt3,
                                                      "Model": self.model_gpt3}
        print("gpt3-small-based-on-gpt2 로드 완료")

        from transformers import ElectraTokenizerFast, ElectraForQuestionAnswering
        self.electra_tokenizer_QA = ElectraTokenizerFast.from_pretrained(
            "monologg/koelectra-base-v3-finetuned-korquad")
        self.electra_model_QA = ElectraForQuestionAnswering.from_pretrained(
            "monologg/koelectra-base-v3-finetuned-korquad")
        self.lists["electra-kor-QA"] = {"Tokenizer": self.electra_tokenizer_QA,
                                        "Model": self.electra_model_QA}

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


models = Models()
