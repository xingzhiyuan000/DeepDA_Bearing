import torch
from torch import nn
import torch.nn.functional as F
# from torchsummary import summary


# 搭建神经网络
class DTLCNN(nn.Module):
    def __init__(self):
        super(DTLCNN, self).__init__()
        # 1 28 28------16 24 24
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0, dilation=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        # 16 24 24------16 12 12
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # 16 12 12------32 6 6
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=0, dilation=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        # 32 6 6------32 3 3
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # 将数据展平
        self.flatten = nn.Flatten()
        # 1024-----64
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.fc1 = nn.Linear(in_features=32 * 3 * 3, out_features=100,bias=False)

        self._feature_dim = 100

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        return x

    def output_num(self):
        return self._feature_dim

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model=DTLCNN().to(device) #将模型转移到cuda上
    input=torch.ones((64,1,28,28)) #生成一个batchsize为64的，1个通道的10X10 tensor矩阵-可以用来检查网络输出大小
    input=input.to(device) #将数据转移到cuda上
    output=Model(input) #将输入喂入网络中进行处理
    print(output.shape)
    # summary(Model,input_size=(1,28,28)) #输入一个通道为3的10X10数据，并展示出网络模型结构和参数
