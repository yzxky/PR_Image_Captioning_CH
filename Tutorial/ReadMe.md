# Tutorial Tips
## 需要注意的事情
+ 若要用cuda加速，需要把网络和数据用.cuda()放到GPU上
+ Pytorch的LSTM需要输入3D的tensor，分别为[序列本身;mini-batch的序号;输入的elements]
+ bp需要.Variable, 且不能打破Variable chain
+ Pytorch是按行映射的，y=Ax+b中将x的行向量映射为y的行向量

## 用到的一些函数
+ (按行)拼接 **torch.cat()**, torch.cat([], axis)
+ reshape tensor **.view()**
+ 激活函数tanh、sigma、ReLU
+ 输出分类 **.softmax(),._log_softmax()**

## 要看的Tutorial
路径：/home/xky/mygit/PR_Image_Captioning_CH/Tutorial

建议阅读顺序：
+ [了解tensor基本概念](http://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html#sphx-glr-beginner-nlp-pytorch-tutorial-py) pytorch_tutorial.py
+ [了解如何搭建网络](http://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html) deep_learning_tutorial.py
+ [了解LSTM的用法](http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) sequence_models_tutorial.py
+ [翻译demo->超有用](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) seq2seq_translation_tutorial.py

选修[Embending](http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

