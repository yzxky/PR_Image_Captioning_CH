# -*- encoding=utf-8 -*-
from __future__ import unicode_literals
import codecs

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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


lang = Lang()

with codecs.open("train_cut.txt", encoding="utf-8") as f:
    for line in f:
        if line.isdigit():
            continue
        lang.addSentence(line)

print lang.index2word[34]
print lang.word2index["一个"]
print lang.word2count["一个"]
print lang.n_words
