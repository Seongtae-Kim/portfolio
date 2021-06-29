class Corpora:
    def __init__(self, dictionary,
                 sejong_path="./corpus/sejong/Sejong_sents_with_senses.pkl",
                 modu_path="./corpus/modu_sense/modu_sents_with_senses.pkl"
                 ):
        #from eunjeon import Mecab
        self.dictionary = dictionary
        #self.tagger = Mecab()
        self.modu = Modu(modu_path)
        self.sejong = Sejong(sejong_path)
        self.concated = self.build_concat()

    def get_sentence_by_lexeme(self, corpus, target_lexeme, target_scode=-1):
        results = []
        for i in corpus:
            for s in corpus[i]["senses"]:
                lexeme = corpus[i]["senses"][s]["lexeme"]
                try:
                    scode = corpus[i]["senses"][s]["scode"]
                except KeyError:
                    scode = None
                if target_lexeme == lexeme and target_scode == -1:
                    results.append({"sent_id": i, "lexeme": lexeme,
                                    "scode": scode, "sentence": corpus[i]["sentence"]})
                elif target_lexeme == lexeme and target_scode == scode:
                    results.append({"sent_id": i, "lexeme": lexeme,
                                    "scode": scode, "sentence": corpus[i]["sentence"]})
        return results

    def build_surface_form_by_senses(self, corpus, sent_id, words_in_dict, sclass_dic=None, with_sclass=False):
        import re
        sent = corpus[sent_id]["sentence"]
        senses = corpus[sent_id]["senses"]
        processed_sent = []
        later_sent = ""
        for i in senses:
            lex = senses[i]["lexeme"]
            try:
                scode = senses[i]["scode"]
            except Exception:
                scode = None

            if not (lex, scode) in words_in_dict:
                continue

            parts = [e for e in re.split("({})".format(
                lex), sent) if e != "" and e != " "]

            if len(parts) == 1 and " " in lex:
                results = self.remove_multiple_occurence_from_string(lex)
                
                for possible_word in results:
                    if re.search(possible_word, parts[0]):
                        lex = possible_word
                        parts = re.split("({})".format(
                            possible_word), parts[0])
                        break

            if len(parts) == 3:
                prior_sent, middle_sent, later_sent = parts
                if with_sclass:
                    if isinstance(sclass_dic[lex][scode], str):
                        sclass = sclass_dic[lex][scode]
                    else:
                        return None
                    processed_sent.extend(
                        [prior_sent, re.sub(lex, "@"+middle_sent+"@ -"+sclass+" ", middle_sent)])  # FROM HERE
                else:
                    processed_sent.extend(
                        [prior_sent, re.sub(lex, "@"+middle_sent+"@ ", middle_sent)])
                sent = later_sent

            elif len(parts) == 2:
                prior_sent, later_sent = parts
                results = self.remove_multiple_occurence_from_string(lex)
                if " " in lex:
                    for possible_word in results:
                        if re.search(possible_word, prior_sent):
                            if with_sclass:
                                if isinstance(sclass_dic[lex][scode], str):
                                    sclass = sclass_dic[lex][scode]
                                else:
                                    return None
                                prior_sent = re.sub(
                                    lex, "@"+possible_word+"@ -"+sclass+" ", prior_sent)
                            else:
                                prior_sent = re.sub(
                                    lex, "@"+possible_word+"@ ", prior_sent)
                            break
                else:
                    if with_sclass:
                        if isinstance(sclass_dic[lex][scode], str):
                            sclass = sclass_dic[lex][scode]
                        else:
                            return None
                        prior_sent = re.sub(
                            lex, "@"+lex+"@ -"+sclass+" ", prior_sent)
                    else:
                        prior_sent = re.sub(lex, "@"+lex+"@ ", prior_sent)

                processed_sent.append(prior_sent)
                sent = later_sent
        processed_sent.append(later_sent)

        return "".join(processed_sent)

    def build_concat(self):
        concat = {}
        n = 0
        for i in self.modu.corpus:
            concat[n] = self.modu.corpus[i]
            n += 1
        for i in self.sejong.corpus:
            concat[n] = self.sejong.corpus[i]
            n += 1
        print(
            "CORPUS: modu+sejong concatenated: {:,} sentences".format(len(concat)))
        return concat

    def remove_multiple_occurence_from_string(self, string, removal=" "):
        from itertools import permutations
        import re
        matches = [m.span() for m in re.finditer(removal, string)]

        combs = []
        for i in range(1, len(matches)+1):
            for comb in list(permutations(matches, i)):
                combs.append(sorted(comb))
        all_possibilities = set()
        for comb in combs:
            all_possibilities.add(self.remove_character(string, comb))
        return all_possibilities

    def remove_character(self, string, matches):
        i = 0
        for match in matches:
            b = match[0]
            e = match[1]
            string = string[:b-i]+string[e-i:]
            i += 1
        return string


class Modu:
    def __init__(self, path):
        self.build_corpus(path)

    def build_corpus(self, path):
        import pickle
        self.corpus = pickle.load(open(path, "rb"))
        print("CORPUS: MODU built: {:,} sentences".format(len(self.corpus)))


class Sejong:
    def __init__(self, path):
        self.build_corpus(path)

    def build_corpus(self, path):
        import pickle
        self.corpus = pickle.load(open(path, "rb"))
        print("CORPUS: Sejong built: {:,} sentences".format(len(self.corpus)))