# encoding=utf-8
import jieba
import codecs

data = []
with codecs.open("train.txt", encoding="utf-8") as f:
    for line in f:
        line = " ".join(jieba.cut(line))
        print line
        data.append(line.strip() + "\n")

with codecs.open("train_cut.txt", "w", encoding="utf-8") as f:
    f.writelines(data)

data = []
with codecs.open("valid.txt", encoding="utf-8") as f:
    for line in f:
        line = " ".join(jieba.cut(line))
        print line
        data.append(line.strip() + "\n")

with codecs.open("valid_cut.txt", "w", encoding="utf-8") as f:
    f.writelines(data)