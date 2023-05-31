import torch.nn as nn
from torchvision import models
from nets.densenet import densenet_bearing
from nets.ghostnet import ghostnet
from nets.mobilenet_v1 import MobileNetV1
import torch.nn.functional as F

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "densenet" == name.lower():
        return DensenetBackbone()
    elif "ghostnet" == name.lower():
        return ghostnet()
    elif "mobilenetv1" == name.lower():
        return MobileV1Backbone()

class DensenetBackbone(nn.Module):
    def __init__(self):
        super(DensenetBackbone, self).__init__()
        model_densenet = densenet_bearing(pretrained=False)
        self.features = model_densenet.features
        self._feature_dim = 448

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

class GhostBackbone(nn.Module):
    def __init__(self):
        super(GhostBackbone, self).__init__()
        model_ghonst = ghostnet()

        self.conv_stem=model_ghonst.conv_stem
        self.bn1=model_ghonst.bn1
        self.act1=model_ghonst.act1
        self.blocks = model_ghonst.blocks

        self._feature_dim = 120

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

class MobileV1Backbone(nn.Module):
    def __init__(self):
        super(MobileV1Backbone, self).__init__()
        model_mobilev1 = MobileNetV1()
        self.stage1 = model_mobilev1.stage1

        self._feature_dim = 256

    def forward(self, x):
        x = self.stage1(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim

if __name__ == "__main__":
    import torch
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = MobileNetV1().to(device)
    # print(m)
    summary(m, input_size=(3, 32, 32))