# 需要注意的事情
+ 若要用cuda加速，需要把网络和数据用.cuda()放到GPU上
+ Pytorch的LSTM需要输入3D的tensor，分别为[序列本身;mini-batch的序号;输入的elements]
