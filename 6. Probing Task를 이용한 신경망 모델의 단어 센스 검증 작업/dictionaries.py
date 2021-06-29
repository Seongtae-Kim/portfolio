class Dictionary:  # for UWIN dataset
    # Seongtae Kim - 2020-12-16
    def __init__(self, uwin_path="/home/seongtae/SynologyDrive/SIRE/Projects/Datasets/UWIN/uwin.dat",
                 urimal_path="./dictionary/urimal_sam.pkl",
                 urimal_sense_path="./dictionary/Urimal_Sam_sense_dic.pkl"):
        self.uwin = UWIN(uwin_path)
        self.urimal = Urimal_SAM(urimal_path, urimal_sense_path)
        self.senses_dic = {'방법', '기준', '존재', '힘', '물질', '관계', '구조물', '공간', '상태', '모양', '활동', '단위', '기기', '행위', '시간', '과정',
                           '정도', '표시', '생물', '재료', '성질', '대상', '구조', '요소', '기호', '범위', '인식', '작용', '인지', '종', '성분', '집단', '종류', '현상', '물건'}


class Urimal_SAM():  # Seongtae Kim - 2021-01-18
    def __init__(self, dictionary_path, sense_path):
        import pickle
        self.words = pickle.load(open(dictionary_path, "rb"))
        print("URIMAL_SAM: Total words in dictionary: {:,}".format(
            len(self.words)))
        self.sense_dic = pickle.load(open(sense_path, "rb"))

        print("URIMAL_SAM: Total senses in dictionary: {:,}".format(
            len(self.sense_dic)))
        print("URIMAL_SAM built")
        print()

    def exist_in_dict(self, target_word, target_scode):
        for i in self.words:
            word = self.words[i]["어휘"]
            scode = self.words[i]["의미 번호"]
            if target_word == word and scode == target_scode:
                return True
        return False

    def get_urimal_definition_by_lexeme(self, target_word, quiet=False, get_senses=True):
        results = []
        for i in self.words:
            word = self.words[i]["어휘"]
            if target_word == word:
                results.append(self.words[i])
        if not quiet:
            print("{} lexemes found".format(len(results)))
            print("senses from lexeme {}".format(target_word))
            if len(results) != 0:
                print(self.sense_dic[target_word])

        if get_senses and len(results) != 0:
            return results, self.sense_dic[target_word]
        else:
            return results, None


class UWIN():  # Seongtae Kim - 2020-12-16
    def __init__(self, path):
        self.words = {}
        self.file = open(path, "r").readlines()
        print("UWIN: Total definitions in dictionary: {:,}".format(
            len(self.file)))
        self.build_uwin_dict()
        print("UWIN built")
        print()

    def build_uwin_dict(self):
        import re
        index_exp = "(^\d+(?=~)|\n\d+(?=~))"
        lexeme_exp = "(?<=~)[가-힣ㄱ-ㅎ]+(?=~)"
        sense_exp = "(?<=~)[가-힣ㄱ-ㅎ]+[0-9]+(?=~)"
        pos_exp = "(?<=~)[가-힣ㄱ-ㅎ]+[\ue000]+[가-힣ㄱ-ㅎ]+(?=~)"
        hanja_exp = "[\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff]+"
        hypernym_exp = "(?<=@).+(?=#)"
        hyponym_exp = "(?<=#).+(?=\$)"
        compelete_synonym_exp = "(?<=\$).+(?=\^)"
        synonym_exp = "(?<=\^).+(?=\|)"
        meaning_exp = "(?<=\|).+(?=$|\n)"

        for definition in self.file:
            index = re.findall(index_exp, definition)
            lexemes = re.findall(lexeme_exp, definition)
            senses = re.findall(sense_exp, definition)
            poss = re.findall(pos_exp, definition)
            tmp = []
            for pos in poss:
                tmp.append([token for token in re.split(
                    "[\ue000]", pos) if token != ""])
            poss = tmp
            hanjas = re.findall(hanja_exp, definition)

            hypernyms = re.findall(hypernym_exp, definition)
            tmp = []
            for hypernym in hypernyms:
                hypernym = re.sub("[^\d@,]+", "", hypernym)
                tmp.extend([int(token) for token in re.split(
                    "[,|@]", hypernym) if token != ""])
            hypernyms = tmp

            hyponyms = re.findall(hyponym_exp, definition)
            tmp = []
            for hyponym in hyponyms:
                hyponym = re.sub("[^\d@,]+", "", hyponym)
                tmp.extend([int(token) for token in re.split(
                    "[,|@]", hyponym) if token != ""])
            hyponyms = tmp

            compelete_synonyms = re.findall(compelete_synonym_exp, definition)
            tmp = []
            for compelete_synonym in compelete_synonyms:
                compelete_synonym = re.sub("[^\d@,]+", "", compelete_synonym)
                tmp.extend([int(token) for token in re.split(
                    "[,|@]", compelete_synonym) if token != ""])
            compelete_synonyms = tmp

            synonyms = re.findall(synonym_exp, definition)
            tmp = []
            for synonym in synonyms:
                synonym = re.sub("[^\d@,]+", "", synonym)
                tmp.extend([int(token) for token in re.split(
                    "[,|@]", synonym) if token != ""])
            synonyms = tmp

            meanings = re.findall(meaning_exp, definition)
            self.add_uwin_definition(index=index, lexemes=lexemes, senses=senses, poss=poss, hanjas=hanjas,
                                     hypernyms=hypernyms, hyponyms=hyponyms,
                                     compelete_synonyms=compelete_synonyms, synonyms=synonyms,
                                     meanings=meanings)
        print("UWIN: WordMap Built")
        self.discard_unknown_keys()
        self.discard_unknown_path()

    def add_uwin_definition(self, index="", lexemes="", senses="", poss="", hanjas="", hypernyms="",
                            hyponyms="", compelete_synonyms="", synonyms="", meanings=""):
        import re

        index = int(index[0])
        lexemes = str(lexemes[0]) if len(lexemes) == 1 else lexemes
        senses = str(senses[0]) if len(senses) == 1 else senses
        senseID = int(re.findall("\d+", senses)
                      [0]) if type(senses) == str else None
        poss = str(poss[0]) if len(poss) == 1 else poss
        hanjas = str(hanjas[0]) if len(hanjas) == 1 else hanjas
        hypernyms = int(hypernyms[0]) if len(hypernyms) == 1 else hypernyms
        hyponyms = int(hyponyms[0]) if len(hyponyms) == 1 else hyponyms
        compelete_synonyms = int(compelete_synonyms[0]) if len(
            compelete_synonyms) == 1 else compelete_synonyms
        synonyms = int(synonyms[0]) if len(synonyms) == 1 else synonyms
        meanings = str(meanings[0]) if len(meanings) == 1 else meanings

        lexemes = set(lexemes) if type(lexemes) is not str else lexemes
        senses = set(senses) if type(senses) is not str else senses
        hanjas = set(hanjas) if type(hanjas) is not str else hanjas
        hypernyms = set(hypernyms) if type(hypernyms) is not int else hypernyms
        hyponyms = set(hyponyms) if type(hyponyms) is not int else hyponyms
        compelete_synonyms = set(compelete_synonyms) if type(
            compelete_synonyms) is not int else compelete_synonyms
        synonyms = set(synonyms) if type(synonyms) is not int else synonyms
        meanings = set(meanings) if type(meanings) is not str else meanings

        lexemes = lexemes.pop() if len(lexemes) == 1 and type(lexemes) is set else lexemes
        senses = senses.pop() if len(senses) == 1 and type(senses) is set else senses
        poss = poss.pop() if len(poss) == 1 and type(poss) is set else poss
        hanjas = hanjas.pop() if len(hanjas) == 1 and type(hanjas) is set else hanjas

        if type(hypernyms) is not int:
            hypernyms = int(hypernyms.pop()) if len(
                hypernyms) == 1 else hypernyms
        if type(hyponyms) is not int:
            hyponyms = int(hyponyms.pop()) if len(
                hyponyms) == 1 and type(hyponyms) is not int else hyponyms
        if type(compelete_synonyms) is not int:
            compelete_synonyms = int(compelete_synonyms.pop()) if len(
                compelete_synonyms) == 1 and type(compelete_synonyms) is not int else compelete_synonyms
        if type(synonyms) is not int:
            synonyms = int(synonyms.pop()) if len(
                synonyms) == 1 and type(synonyms) is not int else synonyms
        meanings = int(meanings.pop()) if len(
            meanings) == 1 and type(meanings) is set else meanings

        lexemes = None if len(lexemes) == 0 else lexemes
        senses = None if len(senses) == 0 else senses
        poss = None if len(poss) == 0 else poss
        hanjas = None if len(hanjas) == 0 else hanjas
        if type(hypernyms) is not int:
            hypernyms = None if len(hypernyms) == 0 else hypernyms
        if type(hyponyms) is not int:
            hyponyms = None if len(hyponyms) == 0 else hyponyms
        if type(compelete_synonyms) is not int:
            compelete_synonyms = None if len(
                compelete_synonyms) == 0 else compelete_synonyms
        if type(synonyms) is not int:
            synonyms = None if len(synonyms) == 0 else synonyms
        meanings = None if len(meanings) == 0 else meanings

        self.words[index] = {"index": index, "lexemes": lexemes, "senses": senses, "senseID": senseID, "poss": poss, "hanjas": hanjas,
                             "hypernyms": hypernyms, "hyponyms": hyponyms,
                             "compelete_synonyms": compelete_synonyms, "synonyms": synonyms,
                             "meanings": meanings}

    def get_uwin_definition_by_lexeme(self, query, quiet=False):
        results = []
        if query != "" and len(self.words) != 0:
            for idx in self.words:
                if type(self.words[idx]["lexemes"]) == list:
                    for lexeme in self.words[idx]["lexemes"]:
                        if lexeme == query:
                            results.append(self.words[idx])
                else:
                    if self.words[idx]["lexemes"] == query:
                        results.append(self.words[idx])
        if not quiet:
            print(len(results), "RESULTS found")
            print()
        if len(results) != 0:
            if not quiet:
                for result in results:
                    print(result)
            return results
        else:
            return None

    def get_maximum_num_hypernyms(self, index, l=[]):
        hypers = self.words[index]["hypernyms"]
        print(self.words[index]["lexemes"])
        print(self.words[index]["hypernyms"])
        print()

        if hypers == 0:  # THE ROOT NODE
            return 0  # no more hypernym

        elif type(hypers) is set:
            for hyper in hypers:
                hypers_len = len(hypers)
                output = self.get_maximum_num_hypernyms(hypers, l)
                if type(output) is list:
                    l.extend(output)
                else:
                    l.append(output)
                l.append(hypers_len)
                return max(l)

        elif type(hypers) is int and hypers != 0:
            hypers_len = 1
            output = self.get_maximum_num_hypernyms(hypers, l)
            if type(output) is list:
                l.extend(output)
            else:
                l.append(output)
            l.append(hypers_len)

            return max(l)

    def get_uwin_id_by_lexeme(self, lexeme, quiet=False):
        results = []
        for index in self.words:
            if self.words[index]["lexemes"] == lexeme:
                results.append(index)
        if not quiet:
            print(len(results), " result")
            print(results)
        return results

    def get_uwin_path(self, index):
        hypers = self.words[index]["hypernyms"]
        hypos = self.words[index]["hyponyms"]
        this_word = self.words[index]["lexemes"]

        if hypers == None:  # THE ROOT NODE
            return ["ROOT"]  # no more hypernym
        elif type(hypers) is int:
            output = self.get_uwin_path(hypers)
            output.append(this_word)
            return output

    def discard_unknown_keys(self):
        wordbook = self.words
        toberemoved = []

        for index in wordbook:
            ids = {}
            ids["hypernyms"] = wordbook[index]["hypernyms"]
            ids["hyponyms"] = wordbook[index]["hyponyms"]
            ids["synonyms"] = wordbook[index]["synonyms"]
            ids["compelete_synonyms"] = wordbook[index]["compelete_synonyms"]

            for collection in ids:
                if isinstance(ids[collection], set):
                    for single_id in ids[collection]:
                        try:
                            wordbook[single_id]
                        except KeyError:
                            toberemoved.append((index, collection, single_id))
                elif isinstance(ids[collection], int):
                    single_id = ids[collection]
                    try:
                        wordbook[single_id]
                    except KeyError:
                        toberemoved.append((index, collection, single_id))

        for index, collection, single_id in toberemoved:
            if isinstance(self.words[index][collection], set):
                self.words[index][collection].discard(single_id)
                if len(self.words[index][collection]) == 1:
                    self.words[index][collection] = int(
                        self.words[index][collection].pop())
            else:
                self.words[index][collection] = None

        print("UWIN: {:,} unknown word ids cleared".format(len(toberemoved)))

    def discard_unknown_path(self):
        cnt = 0
        for index in self.words:  # removing invalid hypernyms
            hyper_ids = self.words[index]["hypernyms"]
            if isinstance(hyper_ids, set):
                invalids = hyper_ids.copy()
                for hyper_id in hyper_ids:
                    parent_hypo_ids = self.words[hyper_id]["hyponyms"]
                    if isinstance(parent_hypo_ids, set):
                        for hypo_id in parent_hypo_ids:
                            if index == hypo_id:
                                invalids.discard(hyper_id)

                    elif isinstance(parent_hypo_ids, int):
                        hypo_id = parent_hypo_ids
                        if index == hypo_id:
                            invalids.discard(hyper_id)

            elif isinstance(hyper_ids, int):
                invalids = {hyper_ids}
                hyper_id = hyper_ids
                parent_hypo_ids = self.words[hyper_id]["hyponyms"]
                if isinstance(parent_hypo_ids, set):
                    for hypo_id in parent_hypo_ids:
                        if index == hypo_id:
                            invalids.discard(hyper_id)
                            break

                elif isinstance(parent_hypo_ids, int):
                    hypo_id = parent_hypo_ids
                    if index == hypo_id:
                        invalids.discard(hyper_id)

            if isinstance(hyper_ids, int) and len(invalids) != 0:
                for invalid in invalids:  # invalid path nodes
                    self.words[index]["hypernyms"] = None
                    cnt += 1
            elif isinstance(hyper_ids, set) and len(invalids) != 0:
                for invalid in invalids:  # invalid path nodes
                    self.words[index]["hypernyms"].discard(invalid)
                    cnt += 1
                if len(self.words[index]["hypernyms"]) == 1:
                    self.words[index]["hypernyms"] = int(
                        self.words[index]["hypernyms"].pop())
                elif len(self.words[index]["hyponyms"]) == 0:
                    self.words[index]["hyponyms"] = None

        print("UWIN: {} invalid hypernyms are deleted".format(cnt))
        cnt = 0
        for index in self.words:  # removing invalid hyponyms
            hypo_ids = self.words[index]["hyponyms"]
            if isinstance(hypo_ids, set):
                invalids = hypo_ids.copy()
                for hypo_id in hypo_ids:
                    child_hyper_ids = self.words[hypo_id]["hypernyms"]
                    if isinstance(child_hyper_ids, set):
                        for hyper_id in child_hyper_ids:
                            if index == hyper_id:
                                invalids.discard(hypo_id)

                    elif isinstance(child_hyper_ids, int):
                        hypo_id = child_hyper_ids
                        if index == hypo_id:
                            invalids.discard(hypo_id)

            elif isinstance(hypo_ids, int):
                invalids = {hypo_ids}
                hypo_id = hypo_ids
                child_hyper_ids = self.words[hypo_id]["hypernyms"]
                if isinstance(child_hyper_ids, set):
                    for hyper_ids in child_hyper_ids:
                        if index == hyper_id:
                            invalids.discard(hypo_id)

                elif isinstance(child_hyper_ids, int):
                    hyper_id = child_hyper_ids
                    if hyper_id == index:
                        invalids.discard(hypo_id)

            if isinstance(hypo_ids, int) and len(invalids) != 0:
                for invalid in invalids:  # invalid path nodes
                    self.words[index]["hyponyms"] = None
                    cnt += 1
            elif isinstance(hypo_ids, set) and len(invalids) != 0:
                for invalid in invalids:  # invalid path nodes
                    self.words[index]["hyponyms"].discard(invalid)
                    cnt += 1
                if len(self.words[index]["hyponyms"]) == 1:
                    self.words[index]["hyponyms"] = int(
                        self.words[index]["hyponyms"].pop())
                elif len(self.words[index]["hyponyms"]) == 0:
                    self.words[index]["hyponyms"] = None
        print("UWIN: {} invalid hyponyms are deleted".format(cnt))

        poss = None if len(poss) == 0 else poss
        hanjas = None if len(hanjas) == 0 else hanjas
        if type(hypernyms) is not int:
            hypernyms = None if len(hypernyms) == 0 else hypernyms
        if type(hyponyms) is not int:
            hyponyms = None if len(hyponyms) == 0 else hyponyms
        if type(compelete_synonyms) is not int:
            compelete_synonyms = None if len(
                compelete_synonyms) == 0 else compelete_synonyms
        if type(synonyms) is not int:
            synonyms = None if len(synonyms) == 0 else synonyms
        meanings = None if len(meanings) == 0 else meanings

        self.words[index] = {"index": index, "lexemes": lexemes, "senses": senses, "senseID": senseID, "poss": poss, "hanjas": hanjas,
                             "hypernyms": hypernyms, "hyponyms": hyponyms,
                             "compelete_synonyms": compelete_synonyms, "synonyms": synonyms,
                             "meanings": meanings}

    def get_uwin_definition_by_lexeme(self, query, quiet=False):
        results = []
        if query != "" and len(self.words) != 0:
            for idx in self.words:
                if type(self.words[idx]["lexemes"]) == list:
                    for lexeme in self.words[idx]["lexemes"]:
                        if lexeme == query:
                            results.append(self.words[idx])
                else:
                    if self.words[idx]["lexemes"] == query:
                        results.append(self.words[idx])
        if not quiet:
            print(len(results), "RESULTS found")
            print()
        if len(results) != 0:
            if not quiet:
                for result in results:
                    print(result)
            return results
        else:
            return None

    def get_maximum_num_hypernyms(self, index, l=[]):
        hypers = self.words[index]["hypernyms"]
        print(self.words[index]["lexemes"])
        print(self.words[index]["hypernyms"])
        print()

        if hypers == 0:  # THE ROOT NODE
            return 0  # no more hypernym

        elif type(hypers) is set:
            for hyper in hypers:
                hypers_len = len(hypers)
                output = self.get_maximum_num_hypernyms(hypers, l)
                if type(output) is list:
                    l.extend(output)
                else:
                    l.append(output)
                l.append(hypers_len)
                return max(l)

        elif type(hypers) is int and hypers != 0:
            hypers_len = 1
            output = self.get_maximum_num_hypernyms(hypers, l)
            if type(output) is list:
                l.extend(output)
            else:
                l.append(output)
            l.append(hypers_len)

            return max(l)

    def get_uwin_id_by_lexeme(self, lexeme, quiet=False):
        results = []
        for index in self.words:
            if self.words[index]["lexemes"] == lexeme:
                results.append(index)
        if not quiet:
            print(len(results), " result")
            print(results)
        return results

    def get_uwin_path(self, index):
        hypers = self.words[index]["hypernyms"]
        hypos = self.words[index]["hyponyms"]
        this_word = self.words[index]["lexemes"]

        if hypers == None:  # THE ROOT NODE
            return ["ROOT"]  # no more hypernym
        elif type(hypers) is int:
            output = self.get_uwin_path(hypers)
            output.append(this_word)
            return output

    def discard_unknown_keys(self):
        wordbook = self.words
        toberemoved = []

        for index in wordbook:
            ids = {}
            ids["hypernyms"] = wordbook[index]["hypernyms"]
            ids["hyponyms"] = wordbook[index]["hyponyms"]
            ids["synonyms"] = wordbook[index]["synonyms"]
            ids["compelete_synonyms"] = wordbook[index]["compelete_synonyms"]

            for collection in ids:
                if isinstance(ids[collection], set):
                    for single_id in ids[collection]:
                        try:
                            wordbook[single_id]
                        except KeyError:
                            toberemoved.append((index, collection, single_id))
                elif isinstance(ids[collection], int):
                    single_id = ids[collection]
                    try:
                        wordbook[single_id]
                    except KeyError:
                        toberemoved.append((index, collection, single_id))

        for index, collection, single_id in toberemoved:
            if isinstance(self.words[index][collection], set):
                self.words[index][collection].discard(single_id)
                if len(self.words[index][collection]) == 1:
                    self.words[index][collection] = int(
                        self.words[index][collection].pop())
            else:
                self.words[index][collection] = None

        print("{:,} unknown word ids cleared".format(len(toberemoved)))

    def discard_unknown_path(self):
        cnt = 0
        for index in self.words:  # removing invalid hypernyms
            hyper_ids = self.words[index]["hypernyms"]
            if isinstance(hyper_ids, set):
                invalids = hyper_ids.copy()
                for hyper_id in hyper_ids:
                    parent_hypo_ids = self.words[hyper_id]["hyponyms"]
                    if isinstance(parent_hypo_ids, set):
                        for hypo_id in parent_hypo_ids:
                            if index == hypo_id:
                                invalids.discard(hyper_id)

                    elif isinstance(parent_hypo_ids, int):
                        hypo_id = parent_hypo_ids
                        if index == hypo_id:
                            invalids.discard(hyper_id)

            elif isinstance(hyper_ids, int):
                invalids = {hyper_ids}
                hyper_id = hyper_ids
                parent_hypo_ids = self.words[hyper_id]["hyponyms"]
                if isinstance(parent_hypo_ids, set):
                    for hypo_id in parent_hypo_ids:
                        if index == hypo_id:
                            invalids.discard(hyper_id)
                            break

                elif isinstance(parent_hypo_ids, int):
                    hypo_id = parent_hypo_ids
                    if index == hypo_id:
                        invalids.discard(hyper_id)

            if isinstance(hyper_ids, int) and len(invalids) != 0:
                for invalid in invalids:  # invalid path nodes
                    self.words[index]["hypernyms"] = None
                    cnt += 1
            elif isinstance(hyper_ids, set) and len(invalids) != 0:
                for invalid in invalids:  # invalid path nodes
                    self.words[index]["hypernyms"].discard(invalid)
                    cnt += 1
                if len(self.words[index]["hypernyms"]) == 1:
                    self.words[index]["hypernyms"] = int(
                        self.words[index]["hypernyms"].pop())
                elif len(self.words[index]["hyponyms"]) == 0:
                    self.words[index]["hyponyms"] = None

        print("{} invalid hypernyms are deleted".format(cnt))
        cnt = 0
        for index in self.words:  # removing invalid hyponyms
            hypo_ids = self.words[index]["hyponyms"]
            if isinstance(hypo_ids, set):
                invalids = hypo_ids.copy()
                for hypo_id in hypo_ids:
                    child_hyper_ids = self.words[hypo_id]["hypernyms"]
                    if isinstance(child_hyper_ids, set):
                        for hyper_id in child_hyper_ids:
                            if index == hyper_id:
                                invalids.discard(hypo_id)

                    elif isinstance(child_hyper_ids, int):
                        hypo_id = child_hyper_ids
                        if index == hypo_id:
                            invalids.discard(hypo_id)

            elif isinstance(hypo_ids, int):
                invalids = {hypo_ids}
                hypo_id = hypo_ids
                child_hyper_ids = self.words[hypo_id]["hypernyms"]
                if isinstance(child_hyper_ids, set):
                    for hyper_ids in child_hyper_ids:
                        if index == hyper_id:
                            invalids.discard(hypo_id)

                elif isinstance(child_hyper_ids, int):
                    hyper_id = child_hyper_ids
                    if hyper_id == index:
                        invalids.discard(hypo_id)

            if isinstance(hypo_ids, int) and len(invalids) != 0:
                for invalid in invalids:  # invalid path nodes
                    self.words[index]["hyponyms"] = None
                    cnt += 1
            elif isinstance(hypo_ids, set) and len(invalids) != 0:
                for invalid in invalids:  # invalid path nodes
                    self.words[index]["hyponyms"].discard(invalid)
                    cnt += 1
                if len(self.words[index]["hyponyms"]) == 1:
                    self.words[index]["hyponyms"] = int(
                        self.words[index]["hyponyms"].pop())
                elif len(self.words[index]["hyponyms"]) == 0:
                    self.words[index]["hyponyms"] = None
        print("{} invalid hyponyms are deleted".format(cnt))
