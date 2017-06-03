# -*- encoding=utf-8 -*-
from __future__ import unicode_literals
import codecs
import re

SOS_token = 0
EOS_token = 1


class Lang(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "", 1: "SOS", 2: "EOS", 3: "unknown"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        # delete punctuation, replace uppercase letters with lowercase letters
        # sentence = sentence.decode("utf-8")
        sentence = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+",\
                          "", sentence)
        sentence = sentence.lower()
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
                if line.isdigit():
                    continue
                self.addSentence(line)


#lang = Lang("train_cut.txt")
#lang.all2index()

# with codecs.open("train_cut.txt", encoding="utf-8") as f:
#     for line in f:
#         if line.isdigit():
#             continue
#         lang.addSentence(line)

#print lang.index2word[34]
#print lang.word2index["一个"]
#print lang.word2count["一个"]
#print lang.n_words
