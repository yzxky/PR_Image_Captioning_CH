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

# transform sentence to list
def txt2list(file_name):
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
            lst = [1]
            for word in line.split(' '):
                lst.append(lang.word2index[word])
            idx_lst.append((idx, lst))
    return idx_lst
