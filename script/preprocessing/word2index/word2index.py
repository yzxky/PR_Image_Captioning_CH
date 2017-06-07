# -*- coding=utf-8 -*-
from __future__ import unicode_literals
import codecs

SOS_token = 1
EOS_token = 2


class Lang(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "", 1: "SOS", 2: "EOS", 3: "unknown"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def all2index(self):
        with codecs.open(self.file_name, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line[0] == u"\ufeff":
                    line = line[1:]
                if line.isdigit():
                    continue
                self.addSentence(line)

    def filterWord(self, threshold = 2):
        for wordIdx in range(4, self.n_words):
            word = self.index2word[wordIdx]
            if self.word2count[word] < threshold:
                self.word2count.pop(word)
                self.index2word.pop(wordIdx)
                self.word2index[word] = 3
                # wordIdx -= 1
                # self.n_words -= 1
        self.n_words = len(self.index2word)


def txt2list(file_name, max_length = 0):
    try:
        max_length = int(max_length)
        lang = Lang(file_name)
        lang.all2index()
        idx_lst = []
        with codecs.open(file_name, encoding="utf-8") as f:
            idx = 1
            for line in f:
                line = line.strip()
                if line[0] == u"\ufeff":
                    line = line[1:]
                if line.isdigit():
                    idx = int(line)
                    continue
                lst = []
                for word in line.split(' '):
                    lst.append(lang.word2index[word])
                if max_length > 0:
                    if max_length <= len(lst):
                        del lst[max_length:]
                    else:
                        lst.extend([0]*(max_length-len(lst)))
                idx_lst.append((idx, lst))
        return lang, idx_lst
    except Exception, e:
        print("The max_length is not an integer")
        return None, []

# idx_list = txt2list("train_cut_process", 'a')
