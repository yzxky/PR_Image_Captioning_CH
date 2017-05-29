# Tutorial Tips
## 需要注意的事情
+ 若要用cuda加速，需要把网络和数据用.cuda()放到GPU上
+ Pytorch的LSTM需要输入3D的tensor，分别为[序列本身;mini-batch的序号;输入的elements]
+ bp需要.Variable, 且不能打破Variable chain
+ Pytorch是按行映射的，y=Ax+b中将x的行向量映射为y的行向量

## 用到的一些函数
+ (按行)拼接 **torch.cat()**
+ reshape tensor **.view()**
+ 激活函数tanh、sigma、ReLU
+ 输出分类 **.softmax(),._log_softmax()**
