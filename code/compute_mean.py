import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt
import sys
import os
import argparse

import datasets
import hopenet
import torch.utils.model_zoo as model_zoo

if __name__ == '__main__':
    batch_size = 1

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(224),
                                          transforms.ToTensor()])

    pose_dataset = datasets.Pose_300W_LP('data/300W_LP', 'data/300W_LP/filename_list_filtered.txt',
                                transformations)
    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    print 'Ready to get mean.'

    for i, (images, labels, name) in enumerate(train_loader):
        print images
