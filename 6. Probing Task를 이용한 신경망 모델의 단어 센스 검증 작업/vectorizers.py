
class Vectorizer():
    def __init__(self, mode, model, corpus):
        from transformers import BertModel
        self.mode_name = mode
        self.model_name = model
        self.build_tokenizer()
        self.bert_model_path = "../BERT-pytorch/bert_pytorch/output/kr-bert/model/pytorch_model.bin"
        self.bert_config_path = "../BERT-pytorch/bert_pytorch/output/kr-bert/model/bert_config.json"
        self.initialize_bert_vocabs()
        self.bert_model = BertModel.from_pretrained(
            self.bert_model_path, config=self.bert_config_path)
        self.whole_sents = self.get_whole_sents(corpus)
        self.parameters = {"embed_size": 300, "window": 5, "min_count": 1, "epochs": 10, "random_state": 2,
                           "max_iter": 2000, "activation": "relu", "hidden_layer_sizes": (64, 64, 64), "solver": "adam"}

    def initialize_bert_vocabs(self):
        import json
        bert_config = json.load(open(self.bert_config_path, "r"))
        bert_config["vocab_size"] = 16424
        with open(self.bert_config_path, "w") as f:
            json.dump(bert_config, f)

    # To escape size discrepancy between size of vocabs in BERT Tokenizer and BERT Model
    def set_bert_vocabs(self, corpus):
        import re
        import json

        assert self.model_name == "bert"

        for sent_id in corpus:
            formed_lexs = None
            formed_sclasses = None

            if corpus[sent_id]["formed"] is not None:
                formed_lexs = re.findall(
                    "@[가-힣ㄱ-ㅎ ]+@", corpus[sent_id]["formed"])
                self.tokenizer.add_tokens(formed_lexs)
            if corpus[sent_id]["sense_formed"] is not None:
                formed_sclasses = re.findall(
                    "@[가-힣]+@ -[가-힣]+", corpus[sent_id]["sense_formed"])
                self.tokenizer.add_tokens(formed_sclasses)

        newsize = len(self.tokenizer.vocab)
        bert_config = json.load(open(self.bert_config_path, "r"))
        print("Vocab size: {:,}".format(bert_config["vocab_size"]))
        bert_config["vocab_size"] = newsize
        print("NEW Vocab size: {:,}".format(newsize))

        with open(self.bert_config_path, "w") as f:
            json.dump(bert_config, f)

    def get_whole_sents(self, corpus, with_sense=False):
        result = []
        for i in corpus:
            result.append(self.tokenize(corpus[i]["sentence"], with_sense))
        return result

    def build_tokenizer(self):
        if self.model_name != "bert":
            from eunjeon import Mecab
            self.tokenizer = Mecab()
        else:
            from transformers import BertTokenizer
            self.bert_vocab_path = "../BERT-pytorch/bert_pytorch/output/kr-bert/vocab.txt"
            self.tokenizer = BertTokenizer.from_pretrained(
                self.bert_vocab_path)

    def tokenize(self, sentence, with_sense=False):
        import re
        result = []
        if with_sense:
            sclasses = re.findall("(?<=@ -)[가-힣]+", sentence)
            formed_sclasses = re.findall("@[가-힣]+@ -[가-힣]+", sentence)
            for part in re.split("(@[가-힣]+@ -[가-힣0-9]+)", sentence):
                if not re.search("(@[가-힣]+@ -[가-힣0-9]+)", part):
                    if self.model_name == "bert":
                        result.extend(self.tokenizer.tokenize(part))
                    else:
                        result.extend(self.tokenizer.morphs(part))
                else:
                    result.append(part)
            return result, formed_sclasses, sclasses
        else:
            for part in re.split("(@[가-힣]+@)", sentence):
                if not re.search("(@[가-힣]+@)", part):
                    if self.model_name == "bert":
                        result.extend(self.tokenizer.tokenize(part))
                    else:
                        result.extend(self.tokenizer.morphs(part))
                else:
                    result.append(part)
            return result

    def build_FastText(self):
        from gensim.models import FastText
        ft = FastText(vector_size=300, negative=10, window=5, min_count=1)
        ft.build_vocab(self.whole_sents)
        self.fasttext = ft
        return ft

    # returning multiple vectors

    def get_Word2Vec_vectors(self, processed_sent, processd_sent_with_sense, lexs_processed, scodes, sclasses, formed_sclasses):
        from gensim.models import Word2Vec
        import torch
        import re
        tokenized_sent = self.tokenize(processed_sent)
        if processd_sent_with_sense is not None:
            tokenized_sent_with_sense, formed_sclasses, sclasses = self.tokenize(
                processd_sent_with_sense, with_sense=True)
        else:
            tokenized_sent_with_sense = None

        results = []

        for processed_lex, processed_lex_with_sense, scode, sclass in zip(lexs_processed, formed_sclasses, scodes, sclasses):
            lexeme = re.sub("@", "", processed_lex).strip()
            if tokenized_sent_with_sense is not None:
                results.append((torch.Tensor(Word2Vec([tokenized_sent], vector_size=300, window=5, negative=10, min_count=1, sg=1).wv.get_vector(processed_lex).reshape(1, -1)),
                                torch.Tensor(Word2Vec([tokenized_sent_with_sense], vector_size=300, window=5, negative=10, min_count=1, sg=1).wv.get_vector(
                                    processed_lex_with_sense).reshape(1, -1)),
                                lexeme, scode, sclass))
            else:
                results.append((torch.Tensor(Word2Vec([tokenized_sent], vector_size=300, window=5, negative=10, min_count=1, sg=1).wv.get_vector(processed_lex).reshape(1, -1)),
                                None,
                                lexeme, scode, None))
        return results

    def get_FastText_vectors(self, processed_sent, processd_sent_with_sense, lexs_processed, scodes, sclasses, formed_sclasses):
        import torch
        import re
        ft = self.ft_model
        tokenized_sent = self.tokenize(processed_sent)
        if processd_sent_with_sense is not None:
            tokenized_sent_with_sense, formed_sclasses, sclasses = self.tokenize(
                processd_sent_with_sense, with_sense=True)
        else:
            tokenized_sent_with_sense = None

        ft.train([tokenized_sent], total_examples=1, epochs=10)
        results = []
        for processed_lex, processed_lex_with_sense, scode, sclass in zip(lexs_processed, formed_sclasses, scodes, sclasses):
            lexeme = re.sub("@", "", processed_lex).strip()

            if tokenized_sent_with_sense is not None:
                results.append((torch.Tensor(ft.wv.get_vector(processed_lex).reshape(1, -1)),
                                torch.Tensor(ft.wv.get_vector(
                                    processed_lex_with_sense).reshape(1, -1)),
                                lexeme,
                                scode, sclass))
            else:
                results.append((torch.Tensor(ft.wv.get_vector(processed_lex).reshape(1, -1)),
                                None,
                                lexeme,
                                scode, None))
        return results

    # targ => to save memory
    def get_BERT_vectors(self, processed_sent, processd_sent_with_sense, lexs_processed, scodes, sclasses, formed_sclasses, targ=["lex_vec", "sense_vec"]):
        import torch
        import re

        output = None
        output_sense = None

        tokenized_sent = self.tokenize(processed_sent)
        if processd_sent_with_sense is not None:
            tokenized_sent_with_sense, formed_sclasses, sclasses = self.tokenize(
                processd_sent_with_sense, with_sense=True)
        else:
            tokenized_sent_with_sense = None

        if len(tokenized_sent) == 0:
            return [None]

        if "lex_vec" in targ:
            encoded = self.tokenizer.encode_plus(tokenized_sent, add_special_tokens=False,
                                                 max_length=230,
                                                 pad_to_max_length=True,
                                                 truncation=True,
                                                 return_tensors='pt')

            input_ids = encoded["input_ids"]

            with torch.no_grad():
                output = self.bert_model(input_ids)[0]

        if "sense_vec" in targ:
            if tokenized_sent_with_sense is not None:
                encoded_sense = self.tokenizer.encode_plus(tokenized_sent_with_sense, add_special_tokens=False,
                                                           max_length=230,
                                                           pad_to_max_length=True,
                                                           truncation=True,
                                                           return_tensors='pt')
                input_ids_sense = encoded_sense["input_ids"]

                with torch.no_grad():
                    output_sense = self.bert_model(input_ids_sense)[0]

        results = []

        if "lex_vec" in targ and "sense_vec" in targ:
            for processed_lex, processed_lex_with_sense, scode, sclass in zip(lexs_processed, formed_sclasses, scodes, sclasses):
                i = tokenized_sent.index(processed_lex)

                lexeme = re.sub("@", "", processed_lex).strip()
                # the vector of word i
                if tokenized_sent_with_sense is not None and output_sense is not None:
                    j = tokenized_sent_with_sense.index(
                        processed_lex_with_sense)
                    results.append(
                        (output[:, i, :], output_sense[:, j, :], lexeme, scode, sclass))
                else:
                    results.append(
                        (None, None, lexeme, scode, None))
            return results

        if "lex_vec" in targ:
            for processed_lex, processed_lex_with_sense, scode, sclass in zip(lexs_processed, formed_sclasses, scodes, sclasses):
                lexeme = re.sub("@", "", processed_lex).strip()
                if tokenized_sent_with_sense is not None and output_sense is not None:
                    i = tokenized_sent.index(processed_lex)
                    # the vector of word i
                    results.append(
                        (output[:, i, :], None, lexeme, scode, None))
                else:
                    results.append((None, None, lexeme, scode, None))
            return results

        if "sense_vec" in targ:
            for processed_lex, processed_lex_with_sense, scode, sclass in zip(lexs_processed, formed_sclasses, scodes, sclasses):
                i = tokenized_sent.index(processed_lex)

                lexeme = re.sub("@", "", processed_lex).strip()
                # the vector of word i
                if tokenized_sent_with_sense is not None and output_sense is not None:
                    j = tokenized_sent_with_sense.index(
                        processed_lex_with_sense)
                    results.append(
                        (None, output_sense[:, j, :], lexeme, scode, sclass))
                else:
                    results.append((None, None, lexeme, scode, None))
            return results

    # returning single vector

    def get_Word2Vec_vector(self, processed_sent, lex_processed, scode):
        from gensim.models import Word2Vec
        import torch
        import re
        tokenized_sent = self.tokenize(processed_sent)
        lexeme = re.sub("@", "", lex_processed).strip()
        return (torch.Tensor(Word2Vec([tokenized_sent], vector_size=300, window=5, negative=10, min_count=1, sg=1).wv.get_vector(lex_processed).reshape(1, -1)),
                lexeme,
                scode)

    def get_FastText_vector(self, processed_sent, lex_processed, scode):
        import torch
        import re
        ft = self.ft_model
        tokenized_sent = self.tokenize(processed_sent)
        ft.train(tokenized_sent, total_examples=1, epochs=1)
        lexeme = re.sub("@", "", lex_processed).strip()
        return (torch.Tensor(ft.wv.get_vector(lex_processed).reshape(1, -1)),
                lexeme,
                scode)

    def get_BERT_vector(self, processed_sent, lex_processed, scode):
        import torch
        import re
        tokenized_sent = self.tokenize(processed_sent)
        if len(tokenized_sent) == 0:
            return [None]
        encoded = self.tokenizer.encode_plus(tokenized_sent, add_special_tokens=False,
                                             max_length=len(tokenized_sent),
                                             pad_to_max_length=True,
                                             truncation=True,
                                             return_tensors='pt')
        input_ids = encoded["input_ids"]
        with torch.no_grad():
            output = self.bert_model(torch.tensor(input_ids))[0]
        i = tokenized_sent.index(lex_processed)
        lexeme = re.sub("@", "", lex_processed).strip()
        return (output[:, i, :],  # only returns [1, 768]
                lexeme,
                scode
                )


class General_Vectorizer(Vectorizer):
    def __init__(self, mode, model, corpus):
        super().__init__(mode, model, corpus)
        if self.model_name == "ft":
            self.ft_model = self.build_FastText()


class Alternative_Vectorizer(Vectorizer):
    def __init__(self, mode, model, corpus):
        super().__init__(mode, model, corpus)
        if self.model_name == "ft":
            self.ft_model = self.build_FastText()

    def uniform_sum(self, vectors):
        assert isinstance(vectors, list)
        summed_vector = None

        for v in vectors:
            if summed_vector is None:
                summed_vector = v.copy()
            else:
                summed_vector += v

        return summed_vector  # summed vector

    # INPUT
    # vectors_with_lexemes => [(vector, lexeme, scode), ...]
    def weighted_sum(self, vector_list, frequencies):
        import numpy as np

        weighted_summed_vector = None
        for v, word, scode in vector_list:
            a = self.calc_alpha(word, scode, frequencies)
            assert a != None

            if weighted_summed_vector is None:
                weighted_summed_vector = v.copy() * a
            else:
                weighted_summed_vector += (a*v)
        return weighted_summed_vector

    # get_lexeme_with_scode_frequency for weighted vector
    def calc_alpha(self, target_lexeme, target_scode,
                   frequencies, with_calc=True):
        if target_lexeme in frequencies:
            if target_scode in frequencies[target_lexeme]:
                target_freq = frequencies[target_lexeme][target_scode]
            else:
                return None
        else:
            return None

        non_target_freq = 0
        for scode in frequencies[target_lexeme]:
            if scode != target_scode:
                non_target_freq += frequencies[target_lexeme][scode]

        if non_target_freq == 0:
            non_target_freq = 1

        if not with_calc:
            return target_freq, non_target_freq
        else:
            return target_freq / non_target_freq
