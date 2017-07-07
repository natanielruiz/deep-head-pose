import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import cv2
import matplotlib.pyplot as plt
import sys
import os
import argparse

import datasets
import hopenet
import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = os.path.join('output/snapshots', args.snapshot + '.pkl')

    model = torchvision.models.resnet18()
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)

    print 'Loading snapshot.'
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.RandomCrop(224), transforms.ToTensor()])

    pose_dataset = datasets.AFLW2000_binned(args.data_dir, args.filename_list,
                                transformations)
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               num_workers=2)

    model.cuda(gpu)

    print 'Ready to test network.'

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    yaw_correct = 0
    pitch_correct = 0
    roll_correct = 0
    total = 0
    for i, (images, labels, name) in enumerate(test_loader):
        images = Variable(images).cuda(gpu)
        labels = Variable(labels).cuda(gpu)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # TODO: There are more efficient ways.
        yaw_correct += (outputs[:][0] == labels[:][0])
        pitch_correct += (outputs[:][])
        for idx in xrange(len(outputs)):
            yaw_correct += (outputs[idx].data[0] == labels[idx].data[0])
            pitch_correct += (outputs[idx].data[1] == labels[idx].data[1])
            roll_correct += (outputs[idx].data[2] == labels[idx].data[2])


    print('Test accuracies of the model on the ' + str(total) +
    ' test images. Yaw: %.4f %%, Pitch: %.4f %%, Roll: %.4f %%' % (yaw_correct / total,
    pitch_correct / total, roll_correct / total))
