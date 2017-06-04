# -*- encoding=utf-8 -*-
from __future__ import unicode_literals
import codecs
import re

data = []
with codecs.open("train_cut.txt", encoding="utf-8") as f:
    for line in f:
        if not line.strip().isdigit():
            # delete punctuation, replace uppercase letters with lowercase letters
            line = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", \
                            "", line)
            line = line.lower()
        data.append(line.strip() + "\n")

with codecs.open("train_cut_process.txt", "w", encoding="utf-8") as f:
    f.writelines(data)

data = []
with codecs.open("valid_cut.txt", encoding="utf-8") as f:
    for line in f:
        if not line.strip().isdigit():
            line = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", \
                            "", line)
            line = line.lower()
        data.append(line.strip() + "\n")

with codecs.open("valid_cut_process.txt", "w", encoding="utf-8") as f:
    f.writelines(data)

data = []
with codecs.open("train_single.txt", encoding="utf-8") as f:
    for line in f:
        if not line.strip().isdigit():
            line = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", \
                            "", line)
            line = line.lower()
        data.append(line.strip() + "\n")

with codecs.open("train_single_process.txt", "w", encoding="utf-8") as f:
    f.writelines(data)

data = []
with codecs.open("valid_single.txt", encoding="utf-8") as f:
    for line in f:
        if not line.strip().isdigit():
            line = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", \
                            "", line)
            line = line.lower()
        data.append(line.strip() + "\n")

with codecs.open("valid_single_process.txt", "w", encoding="utf-8") as f:
    f.writelines(data)