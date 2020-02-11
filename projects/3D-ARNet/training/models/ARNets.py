import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet


def Conv_bn(inp, oup, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class Inception_pool(nn.Module):
    def __init__(self):
        super(Inception_pool, self).__init__()
        self.pool0 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pool1 = nn.AdaptiveAvgPool3d((2, 1, 1))
        self.pool2 = nn.AdaptiveAvgPool3d((3, 1, 1))

    def forward(self, x):
        x0 = self.pool0(x)
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        out = torch.cat((x0, x1, x2), 2)
        return out


class Inception_block_spectrum(nn.Module):
    def __init__(self, inplanes, planes):
        super(Inception_block_spectrum, self).__init__()
        inter_planes = planes//8
        self.branch0 = nn.Sequential(
            Conv_bn(inplanes, inter_planes*2, kernel_size=1, stride=1)
        )
        self.branch1 = nn.Sequential(
            Conv_bn(inplanes, inter_planes*2, kernel_size=1, stride=1),
            Conv_bn(inter_planes*2, inter_planes*4, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        )
        self.branch2 = nn.Sequential(
            Conv_bn(inplanes, inter_planes, kernel_size=1, stride=1),
            Conv_bn(inter_planes, inter_planes*2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            Conv_bn(inter_planes*2, inter_planes*2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        )
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out += x
        out = self.relu(out)
        return out


class Inception_block_space(nn.Module):
    def __init__(self, inplanes, planes):
        super(Inception_block_space, self).__init__()
        self.downsample = None
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm3d(planes),
            )
        inter_planes = planes//8
        self.branch0 = nn.Sequential(
            Conv_bn(inplanes, inter_planes*2, kernel_size=1, stride=1)
        )
        self.branch1 = nn.Sequential(
            Conv_bn(inplanes, inter_planes*2, kernel_size=1, stride=1),
            Conv_bn(inter_planes*2, inter_planes*4, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        )
        self.branch2 = nn.Sequential(
            Conv_bn(inplanes, inter_planes, kernel_size=1, stride=1),
            Conv_bn(inter_planes, inter_planes*2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            Conv_bn(inter_planes*2, inter_planes*2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        )
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        residual = x
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ARNet(nn.Module):
    def __init__(self, layers, num_classes=200, dropout_keep_prob=0):
        self.inplanes = 32
        super(ARNet, self).__init__()
        self.conv = nn.Sequential(
            Conv_bn(1, 32, kernel_size=(8, 3, 3), stride=1, padding=0),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Inception_block_space(32, 32),
            self._make_layer(Inception_block_spectrum, 32, layers[0]),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Inception_block_space(32, 64),
            self._make_layer(Inception_block_spectrum, 64, layers[1]),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Inception_block_space(64, 128),
            self._make_layer(Inception_block_spectrum, 128, layers[2]),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Inception_block_space(128, 256),
            self._make_layer(Inception_block_spectrum, 256, layers[3]),
        )
        self.pool = Inception_pool()
        self.classifier = nn.Conv3d(256, num_classes, kernel_size=(1, 1, 1))
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks):
        layers = []
        layers.append(block(planes, planes))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.classifier(x)
        x = self.global_pool(x)
        x = x.squeeze(2).squeeze(2).squeeze(2)

        return F.log_softmax(x, dim=1)


def ARNet_1(**kwargs):
    model = ARNet([1, 1, 1, 1], **kwargs)
    return model


def ARNet_2(**kwargs):
    model = ARNet([1, 2, 2, 1], **kwargs)
    return model

def ARNet_3(**kwargs):
    model = ARNet([2, 2, 2, 2], **kwargs)
    return model

def ARNet_4(**kwargs):
    model = ARNet([3, 4, 6, 3], **kwargs)
    return model

dict={'ARNet_1':ARNet_1, 'ARNet_2':ARNet_2, 'ARNet_3':ARNet_3, 'ARNet_4':ARNet_4}
