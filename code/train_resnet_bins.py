import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn

import cv2
import matplotlib.pyplot as plt
import sys
import os
import argparse

from datasets import Pose_300W_LP
import hopenet

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc_pitch = nn.Linear(num_ftrs, 3)
    model.fc_yaw = nn.Linear(num_ftrs, 3)
    model.fc_roll = nn.Linear(num_ftrs, )

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(230),transforms.RandomCrop(224),
                                          transforms.ToTensor()])

    pose_dataset = Pose_300W_LP(args.data_dir, args.filename_list,
                                transformations)
    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    model.cuda(gpu)
    criterion = nn.MSELoss(size_average = True)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr = args.lr)

    print 'Ready to train network.'

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)
            labels = Variable(labels).cuda(gpu)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size, loss.data[0]))

        # Save models at even numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs - 1:
            print 'Taking snapshot...'
            torch.save(model.state_dict(),
            'output/snapshots/resnet18_epoch_' + str(epoch+1) + '.pkl')

    # Save the final Trained Model
    torch.save(model.state_dict(), 'output/snapshots/resnet18_epoch_' + str(epoch+1) + '.pkl')
