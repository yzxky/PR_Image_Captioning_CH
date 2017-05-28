import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()#init error occurs without this
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#init a net
net = Net()
net = net.cuda()
print(net)

# input a picture of 32*32(why nSamples*nChannels*Height*Width)
input = Variable(torch.randn(1, 1, 32, 32))
input = input.cuda()

# torch.cuda.is_available() can test the cuda state
# build the target
temp = torch.zeros(10)
temp[3] = 1
target = Variable(temp.cuda())  # a dummy target, for example
criterion = nn.MSELoss()

# update weights(Stochastic Gradient Descent for example)
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
# in your training loop:
for loop in range(1,10000):
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update
    print("Loop Num: %d, loss = %f" % (loop,loss.data[0]))
    if loss.data[0] < 1e-4:
        break
# your final loss is a cuda tensor
print(loss)

