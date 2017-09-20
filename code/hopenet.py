import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

# CNN Model (2 conv layer)
class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(17*17*512, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class Hopenet(nn.Module):
    # This is just Hopenet with 3 output layers for yaw, pitch and roll.
    def __init__(self, block, layers, num_bins, iter_ref):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        self.softmax = nn.Softmax()
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        self.idx_tensor = Variable(torch.FloatTensor(range(66))).cuda()

        self.iter_ref = iter_ref

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        yaw = self.softmax(pre_yaw)
        yaw = Variable(torch.sum(yaw.data * self.idx_tensor.data, 1), requires_grad=True)
        pitch = self.softmax(pre_pitch)
        pitch = Variable(torch.sum(pitch.data * self.idx_tensor.data, 1), requires_grad=True)
        roll = self.softmax(pre_roll)
        roll = Variable(torch.sum(roll.data * self.idx_tensor.data, 1), requires_grad=True)
        yaw = yaw.view(yaw.size(0), 1)
        pitch = pitch.view(pitch.size(0), 1)
        roll = roll.view(roll.size(0), 1)
        angles = []
        preangles = torch.cat([yaw, pitch, roll], 1)
        angles.append(preangles)

        # angles predicts the residual
        for idx in xrange(self.iter_ref):
            angles.append(self.fc_finetune(torch.cat((preangles, x), 1)))

        return pre_yaw, pre_pitch, pre_roll, angles
