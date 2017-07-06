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

from datasets import AFLW2000
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

    transformations = transforms.Compose([transforms.Scale(224),transforms.RandomCrop(224), transforms.ToTensor()])

    pose_dataset = AFLW2000(args.data_dir, args.filename_list,
                                transformations)
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               num_workers=2)

    model.cuda(gpu)

    print 'Ready to test network.'

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    error = .0
    total = 0
    count = 0
    for i, (images, labels, name) in enumerate(test_loader):
        images = Variable(images).cuda(gpu)
        labels = Variable(labels).cuda(gpu)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # TODO: There are more efficient ways.
        for idx in xrange(len(outputs)):
            if abs(labels[idx].data[2]) * 180 / np.pi > 100:
                print name
                count += 1
                # print abs(outputs[idx].data[0] - labels[idx].data[0]) * 180 / np.pi, 180 * outputs[idx].data[0] / np.pi, labels[idx].data[0] * 180 / np.pi
                print labels[idx].data * 180 / np.pi

            # error += utils.mse_loss(outputs[idx], labels[idx])
            error += abs(outputs[idx].data[0] - labels[idx].data[0]) * 180 / np.pi

    print 'count ', count
    print('Test MSE error of the model on the ' + str(total) +
    ' test images: %.4f' % (error / total))
