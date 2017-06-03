# encoding=utf-8
import codecs

data = []
with codecs.open("train.txt", encoding="utf-8") as f:
    for line in f:
        if not line.strip().isdigit():
            line = " ".join(line)
        print line
        data.append(line.strip() + "\n")

with codecs.open("train_single.txt", "w", encoding="utf-8") as f:
    f.writelines(data)

data = []
with codecs.open("valid.txt", encoding="utf-8") as f:
    for line in f:
        if not line.strip().isdigit():
            line = " ".join(line)
        print line
        data.append(line.strip() + "\n")

with codecs.open("valid_single.txt", "w", encoding="utf-8") as f:
    f.writelines(data)