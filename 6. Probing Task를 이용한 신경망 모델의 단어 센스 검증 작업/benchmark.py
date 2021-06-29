class Lexical_Benchmark:
    def __init__(self, uwin_path="../Datasets/UWIN/uwin.dat",
                 urimal_path="./dictionary/urimal_sam.pkl",
                 urimal_sense_path="./dictionary/Urimal_Sam_sense_dic.pkl",
                 sejong_path="./corpus/sejong/Sejong_sents_with_senses.pkl",
                 modu_path="./corpus/modu_sense/modu_sents_with_senses.pkl"
                 ):

        from dictionaries import Dictionary
        from corpora import Corpora

        print("==================================================================")
        self.dictionaries = Dictionary(
            uwin_path, urimal_path, urimal_sense_path)
        print("==================================================================")
        self.map_sense_class()
        self.corpora = Corpora(
            self.dictionaries, sejong_path, modu_path)
        self.build_sense_form(self.corpora.concated)
        # self.build_sense_form(self.corpora.sejong.corpus)
        # self.build_sense_form(self.corpora.modu.corpus)
        print("==================================================================")
        self.wordbook = Word_Dictionary(self)
        print("==================================================================")
        self.build_general_vectorizers()
        self.build_alternative_vectorizers()
        print("==================================================================")
        self.build_general_vectors()
        print("VECTORS: General Vectors Built.")
        self.build_alternative_vectors()
        print("VECTORS: Alternative Vectors Built.")
        print("==================================================================")

    def train_classifiers(self, model, method):
        from tqdm import tqdm

        c = None
        if model == "w2v":
            c = self.w2v_classifiers
        elif model == "ft":
            c = self.ft_classifiers
        elif model == "bert":
            c = self.bert_classifiers
        assert c is not None

        for word in tqdm(c.classifiers):
            if len(c.classifiers[word]) >= 2:  # for sense
                for target_sense_id in c.classifiers[word]:
                    target_mlp = c.classifiers[word][target_sense_id]["mlp"]
                    target_vector = (
                        c.classifiers[word][target_sense_id]["vector"], target_sense_id)

                    non_target_vectors = []
                    for non_target_sense_id in c.classifiers[word]:
                        non_target_vectors.append(
                            (c.classifiers[word][non_target_sense_id]["vector"], non_target_sense_id))

                    if method == "general":
                        vectors, labels = c.get_general_vector(
                            target_vector, non_target_vectors)
                        trained_mlp, acc = c.train(
                            word, target_sense_id, target_mlp, vectors, labels, quiet=True)
                        c.classifiers[word][target_sense_id].update(
                            {"mlp": trained_mlp})
                        c.classifiers[word][target_sense_id].update(
                            {"trained": True})
                        c.classifiers[word][target_sense_id].update(
                            {"accuracy": acc})

                    elif method == "uniform":
                        vector, _ = c.get_uniform_vector(
                            target_vector, non_target_vectors)
                    elif method == "weighted":
                        vector, _ = c.get_weighted_vector(
                            target_vector, non_target_vectors, word, self.wordbook.frequeincies)

    def create_classifiers(self, models=["w2v", "ft", "bert"], initialize=False):
        from classifiers import Classifiers
        import pickle

        if not initialize:
            if "w2v" in models:
                self.w2v_classifiers = pickle.load(
                    open("./classifiers/w2v_classifiers_untrained.pkl", "rb"))
            if "ft" in models:
                self.ft_classifiers = pickle.load(
                    open("./classifiers/ft_classifiers_untrained.pkl", "rb"))
            if "bert" in models:
                self.bert_classifiers = pickle.load(
                    open("./classifiers/bert_classifiers_untrained.pkl", "rb"))
        else:
            if "w2v" in models:
                w2v_vecs = pickle.load(
                    open("./vectors/concated_w2v_general_vectors.pkl", "rb"))
                self.w2v_classifier = Classifiers("w2v")
                for word in w2v_vecs:
                    vector = word[0]

                    lexeme = word[1]
                    scode = word[2]
                    self.w2v_classifier.create_classifier(
                        lexeme, scode, vector)
                with open("./classifiers/w2v_classifiers_untrained.pkl", "wb") as f:
                    pickle.dump(self.w2v_classifier, f)
                print("Word2Vec classifiers saved")

            if "ft" in models:
                ft_vecs = pickle.load(
                    open("./vectors/concated_ft_general_vectors.pkl", "rb"))
                self.ft_classifier = Classifiers("ft")
                for word in ft_vecs:
                    vector = word[0]
                    lexeme = word[1]
                    scode = word[2]
                    self.ft_classifier.create_classifier(lexeme, scode, vector)
                with open("./classifiers/ft_classifiers_untrained.pkl", "wb") as f:
                    pickle.dump(self.ft_classifier, f)
                print("FastText classifiers saved")

            if "bert" in models:
                bert_vecs = pickle.load(
                    open("./vectors/concated_bert_general_vectors.pkl", "rb"))
                self.bert_classifier = Classifiers("bert")
                for word in bert_vecs:
                    if word != None:
                        vector = word[0]
                        lexeme = word[1]
                        scode = word[2]
                        self.bert_classifier.create_classifier(
                            lexeme, scode, vector)
                with open("./classifiers/bert_classifiers_untrained.pkl", "wb") as f:
                    pickle.dump(self.bert_classifier, f)
                print("BERT classifiers saved")

    # Building Vectorizers

    def build_general_vectorizers(self):
        from vectorizers import General_Vectorizer
        self.w2v_general_vectorizer = General_Vectorizer(
            "general", "w2v", self.corpora.concated)
        self.ft_general_vectorizer = General_Vectorizer(
            "general", "ft", self.corpora.concated)
        self.bert_general_vectorizer = General_Vectorizer(
            "general", "bert", self.corpora.concated)
        print("VECTORIZERS: general vectorizers created")

    def build_alternative_vectorizers(self):
        from vectorizers import Alternative_Vectorizer
        self.w2v_alternative_vectorizer = Alternative_Vectorizer(
            "alternative", "w2v", self.corpora.concated)
        self.ft_alternative_vectorizer = Alternative_Vectorizer(
            "alternative", "ft", self.corpora.concated)
        self.bert_alternative_vectorizer = Alternative_Vectorizer(
            "alternative", "bert", self.corpora.concated)
        print("VECTORIZERS: alternative vectorizers created")

    # mode: general / alternative
    # model: w2v / ft / bert
    def create_vectorizer(self, mode, model, corpus):
        from vectorizers import General_Vectorizer, Alternative_Vectorizer

        if mode == "general":
            self.vectorizer = General_Vectorizer(mode, model, corpus)
        elif mode == "alternative":
            self.vectorizer = Alternative_Vectorizer(mode, model, corpus)
        return self.vectorizer

    # Building Vectors

    def build_general_vectors(self, models=["w2v", "ft", "bert"], initialize=False):
        import pickle

        if initialize:
            if "w2v" in models:
                self.w2v_general_vecs = self.corpus2vecs(
                    self.corpora.concated, self.w2v_general_vectorizer)
                with open("./vectors/concated_w2v_general_vectors.pkl", "wb") as f:
                    pickle.dump(self.w2v_general_vecs, f)
            if "ft" in models:
                self.ft_general_vecs = self.corpus2vecs(
                    self.corpora.concated, self.ft_general_vectorizer)
                with open("./vectors/concated_ft_general_vectors.pkl", "wb") as f:
                    pickle.dump(self.ft_general_vecs, f)
            if "bert" in models:
                self.bert_general_vecs = self.corpus2vecs(
                    self.corpora.concated, self.bert_general_vectorizer)
                with open("./vectors/concated_bert_general_vectors.pkl", "wb") as f:
                    pickle.dump(self.bert_general_vecs, f)
        else:
            if "w2v" in models:
                self.w2v_general_vecs = pickle.load(
                    open("./vectors/concated_w2v_general_vectors.pkl", "rb"))
            if "ft" in models:
                self.ft_general_vecs = pickle.load(
                    open("./vectors/concated_ft_general_vectors.pkl", "rb"))
            if "bert" in models:
                self.bert_general_vecs = pickle.load(
                    open("./vectors/concated_bert_general_vectors.pkl", "rb"))

    def build_alternative_vectors(self, models=["w2v", "ft", "bert"], initialize=True):
        import pickle
        if initialize:
            if "w2v" in models:
                self.w2v_general_vecs = pickle.load(
                    open("./vectors/concated_w2v_general_vectors.pkl", "rb"))
            if "ft" in models:
                self.ft_general_vecs = pickle.load(
                    open("./vectors/concated_ft_general_vectors.pkl", "rb"))
            if "bert" in models:
                # self.bert_general_vectorizer.set_bert_vocabs(self.corpora.concated)
                self.bert_general_vecs = pickle.load(
                    open("./vectors/concated_bert_general_vectors.pkl", "rb"))

            self.w2v_alternative_vecs = self.corpus2vecs(
                self.corpora.concated, self.w2v_alternative_vectorizer)
            self.ft_alternative_vecs = self.corpus2vecs(
                self.corpora.concated, self.ft_alternative_vectorizer)
            self.bert_alternative_vecs = self.corpus2vecs(
                self.corpora.concated, self.bert_alternative_vectorizer)

            with open("./vectors/concated_w2v_alternative_vectors.pkl", "wb") as f:
                pickle.dump(self.w2v_alternative_vecs, f)
            with open("./vectors/concated_ft_alternative_vectors.pkl", "wb") as f:
                pickle.dump(self.ft_alternative_vecs, f)
            with open("./vectors/concated_bert_alternative_vectors.pkl", "wb") as f:
                pickle.dump(self.bert_alternative_vecs, f)
        else:
            self.w2v_general_vecs = pickle.load(
                open("./vectors/concated_w2v_alternative_vectors.pkl", "rb"))
            self.ft_general_vecs = pickle.load(
                open("./vectors/concated_ft_alternative_vectors.pkl", "rb"))
            self.bert_general_vecs = pickle.load(
                open("./vectors/concated_bert_alternative_vectors.pkl", "rb"))

    # Convert Corpus to Vectors
    # targ is used only for BERT => to save memory
    def corpus2vecs(self, corpus, vectorizer, targ=["lex_vec", "sense_vec"]):
        from tqdm import tqdm
        vectors = []
        for sent_id in tqdm(corpus):
            formed_sent = corpus[sent_id]["formed"]
            sense_formed_sent = corpus[sent_id]["sense_formed"]
            formed_lexs = self.get_formed_lexemes(corpus, sent_id)
            lexs = [self.formed_lex_to_original(lex) for lex in formed_lexs]
            scodes = self.find_scodes_from_lexs(
                lexs, corpus, sent_id)
            sclasses = [self.get_sclass(l, s) for l, s in zip(lexs, scodes)]
            formed_sclasses = self.get_formed_sclasses(formed_lexs, sclasses)

            assert len(formed_lexs) == len(scodes) == len(sclasses)

            if vectorizer.model_name == "w2v":
                vectors.extend(vectorizer.get_Word2Vec_vectors(
                    formed_sent, sense_formed_sent, formed_lexs, scodes, sclasses, formed_sclasses))
            elif vectorizer.model_name == "ft":
                vectors.extend(vectorizer.get_FastText_vectors(
                    formed_sent, sense_formed_sent, formed_lexs, scodes, sclasses, formed_sclasses))
            elif vectorizer.model_name == "bert":
                vectors.extend(vectorizer.get_BERT_vectors(
                    formed_sent, sense_formed_sent, formed_lexs, scodes, sclasses, formed_sclasses, targ))
        return vectors

    # Looking for slcasses
    def get_sclass(self, lex, scode):
        if lex in self.sense_classes:
            if scode in self.sense_classes[lex]:
                if isinstance(self.sense_classes[lex][scode], str):
                    return self.sense_classes[lex][scode]

        return None

    def get_formed_sclasses(self, formed_lexs, sclasses):
        formed_sclasses = []
        for l, s in zip(formed_lexs, sclasses):
            if l is not None and s is not None:
                formed_sclasses.append("".join([l, " -", s]))
            else:
                formed_sclasses.append(None)
        return formed_sclasses

    # Looking for scodes

    def find_scodes_from_lexs(self, lexs, corpus, sent_id):
        lexemes_found = set()
        scodes = []
        for lex in lexs:
            for i in corpus[sent_id]["senses"]:
                if corpus[sent_id]["senses"][i]["lexeme"] == lex and lex not in lexemes_found:
                    scodes.append(corpus[sent_id]["senses"][i]["scode"])
                    lexemes_found.add(lex)
                    break
        assert len(scodes) == len(lexs)
        return list(scodes)

    def formed_lex_to_original(self, formed_lex):
        import re
        return re.sub("@", "", formed_lex).strip()

    def get_formed_lexemes(self, corpus, sent_id):
        import re
        return re.findall("@[가-힣ㄱ-ㅎ ]+@", corpus[sent_id]["formed"])

    def build_sense_form(self, corpus):
        sdic = self.dictionaries.urimal.sense_dic
        for sent_id in corpus:
            words_in_dict = []
            for sense_id in corpus[sent_id]["senses"]:
                lex = corpus[sent_id]["senses"][sense_id]["lexeme"]
                try:
                    scode = corpus[sent_id]["senses"][sense_id]["scode"]
                except KeyError:
                    scode = "None"
                if lex in sdic:
                    if scode in sdic[lex] or scode in sdic[lex]:
                        words_in_dict.append((lex, scode))

            formed = self.corpora.build_surface_form_by_senses(
                corpus, sent_id, words_in_dict)
            formed_senses = self.corpora.build_surface_form_by_senses(
                corpus, sent_id, words_in_dict, sdic, with_sclass=True) # self.sense_classes
            corpus[sent_id].update(
                {"formed": formed, "sense_formed": formed_senses})

    # Mapping Sense Classes
    def map_sense_class(self, initialize=False):
        from tqdm.notebook import tqdm
        from models import BERT
        import os
        import pickle

        if not initialize:
            self.sense_classes = pickle.load(
                open("./dictionary/sense_classes.pkl", "rb"))

        elif initialize:
            bert = BERT()
            bert.model

            if os.path.exists("./dictionary/checkpoint") and os.path.exists("./dictionary/sense_classes.pkl"):
                checkpoint = int(open("./dictionary/checkpoint", "r").read())
                freq = pickle.load(
                    open("./dictionary/sense_classes.pkl", "rb"))
            else:
                import copy
                freq = copy.deepcopy(self.wordbook.frequeincies)
                checkpoint = 0

            for i in tqdm(self.corpora.concated):
                if i < checkpoint:
                    continue

                for s in self.corpora.concated[i]["senses"]:
                    try:
                        sense = self.corpora.concated[i]["senses"][s]["scode"]

                    except KeyError:
                        continue

                    word = self.corpora.concated[i]["senses"][s]["lexeme"]

                    if self.dictionaries.urimal.exist_in_dict(word, sense):
                        result = self.map_uwin_to_urimal(word, bert)
                        if result is not None:
                            for mapped in result:
                                uwin_id = mapped[1][1]
                                path = self.dictionaries.uwin.get_uwin_path(
                                    uwin_id)
                                if len(path) >= 2:
                                    urimal_sense_class_code = mapped[2][1]
                                    sense_class_name = path[1]
                                    freq[word][urimal_sense_class_code] = sense_class_name

                with open("./dictionary/sense_classes.pkl", "wb") as f:
                    pickle.dump(freq, f)
                with open("./dictionary/checkpoint", "w") as f:
                    f.write(str(i+1))
            with open("./dictionary/sense_classes.pkl", "wb") as f:
                pickle.dump(freq, f)

            self.sense_classes = freq

    # Mapping Sense class by single lexeme
    def map_uwin_to_urimal(self, lexeme, bert):
        from torch.nn import CosineSimilarity
        import torch

        cos = CosineSimilarity()

        uwin_meanings = []
        results = self.dictionaries.uwin.get_uwin_definition_by_lexeme(
            lexeme, quiet=True)
        if results is None:
            return None

        for result in results:
            uwin_meanings.append((result["meanings"], result["index"]))

        urimal_meanings = []
        result = self.dictionaries.urimal.get_urimal_definition_by_lexeme(
            lexeme, quiet=True)[1]
        if result is None:
            return None

        for sense_id in result:
            urimal_meanings.append((result[sense_id], sense_id))
        try:
            # and len(uwin_meanings)<=len(urimal_meanings)
            assert len(uwin_meanings) != 0 and len(urimal_meanings) != 0
        except AssertionError:
            print("UWIN 결과 {}개".format(len(uwin_meanings)))
            print("URIMAL 결과 {}개".format(len(urimal_meanings)))
            raise
        mx = 0
        for sent, _ in uwin_meanings + urimal_meanings:
            length = len(bert.tokenizer.tokenize(sent))
            if length > mx:
                mx = length
        uwin_encoded = [(bert.encode(bert.tokenizer, e, max_length=mx), uwin_id)
                        for e, uwin_id in uwin_meanings]
        urimal_encoded = [(bert.encode(bert.tokenizer, e, max_length=mx), sense_id)
                          for e, sense_id in urimal_meanings]

        uwin_features = [(bert.model(e)[1], uwin_id)
                         for e, uwin_id in uwin_encoded]
        urimal_features = [(bert.model(e)[1], sense_id)
                           for e, sense_id in urimal_encoded]

        answers = []
        j_s = []

        for i, (uwin_f, uwin_id) in enumerate(uwin_features):
            candids = []
            for j, (urimal_f, urimal_sense_id) in enumerate(urimal_features):
                if j not in j_s:
                    output = cos(uwin_f, urimal_f)
                    score = float(torch.mean(output).cpu().detach().numpy())
                    candids.append((score, (i, j)))
            scores = [s for s, _ in candids]
            if len(scores) != 0:
                index = scores.index(max(scores))
                answer = candids[index]
                answers.append(
                    (answer[0], uwin_meanings[answer[1][0]], urimal_meanings[answer[1][1]]))
                j_s.append(answer[1][1])
        return answers

    # get available urimal senses from modu corpus sentence

    def get_available_senses_from_sentence(self, corpus, sent_id):
        #sent = corpus.corpus[sent_id]["sentence"]
        results = []
        for i in corpus.corpus[sent_id]["senses"].keys():
            lexeme = corpus.corpus[sent_id]["senses"][i]["lexeme"]
            try:
                cor_sense_id = corpus.corpus[sent_id]["senses"][i]["scode"]
            except KeyError:
                cor_sense_id = None
            if " " in lexeme:
                for alternative_lexeme in corpus.remove_multiple_occurence_from_string(string=lexeme):
                    if self.check_availability(alternative_lexeme, cor_sense_id):
                        results.append((alternative_lexeme, cor_sense_id))
            else:
                if self.check_availability(lexeme, cor_sense_id):
                    results.append((lexeme, cor_sense_id))
        return results

    # check (lexeme, sense_id) in corpus is available in urimal dictionary
    def check_availability(self, lexeme, cor_sense_id, quiet=True):
        result = self.dictionaries.urimal.get_urimal_definition_by_lexeme(
            lexeme, quiet=True)
        if result != ([], None):
            for dic_sense_id in result[1]:
                if dic_sense_id == cor_sense_id:
                    return True
            return False
        else:
            return False


class Word_Dictionary():
    def __init__(self, benchmark):
        corpus = benchmark.corpora.concated
        urimal = benchmark.dictionaries.urimal
        assert corpus is not None
        import pickle
        self.frequeincies = pickle.load(
            open("./dictionary/matched_words/frequencies.pkl", "rb"))
        #self.frequeincies = self.get_frequencies(corpus, urimal)
        scodes = sum([len(self.frequeincies[l]) for l in self.frequeincies])
        print("WORD_DICTIONARY: {:,} unique lexemes and {:,} unique scodes are assembled (sejong, modu)".format(
            len(self.frequeincies), scodes))

    def get_frequencies(self, corpus, dictionary):
        from tqdm import tqdm
        import pickle
        frequencies = {}
        i = 0
        with tqdm(corpus, leave=False, bar_format="{percentage:2.2f}%{bar} [{elapsed}<{remaining}] | {desc}") as t:
            for sent_id in corpus:
                i += 1
                if i % 100 == 0:
                    with open("./dictionary/matched_words/frequencies.pkl", "wb") as f:
                        pickle.dump(frequencies, f)
                t.update()
                for sense_id in corpus[sent_id]["senses"]:
                    l = corpus[sent_id]["senses"][sense_id]["lexeme"]
                    try:
                        s = corpus[sent_id]["senses"][sense_id]["scode"]
                    except Exception:
                        continue  # some words have no scodes
                    exist = dictionary.exist_in_dict(l, s)
                    if exist:
                        if l in frequencies:
                            if s not in frequencies[l]:
                                frequencies[l].update({s: 1})
                            else:
                                frequencies[l][s] += 1
                        else:
                            frequencies.update({l: {s: 1}})
                    t.set_description_str(
                        "sentence: {:,}/{:,} | lexeme: {} / scode:{} => {}".format(i, len(corpus), l, s, exist))

        with open("./dictionary/matched_words/frequencies.pkl", "wb") as f:
            pickle.dump(frequencies, f)
        return frequencies
