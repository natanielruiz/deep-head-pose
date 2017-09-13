import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

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
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
          default=False, type=bool)
    parser.add_argument('--iter_ref', dest='iter_ref', default=1, type=int)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot

    # ResNet101 with 3 outputs.
    # model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
    # ResNet50
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66, args.iter_ref)
    # ResNet18
    # model = hopenet.Hopenet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)

    print 'Loading snapshot.'
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list,
                                transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    else:
        print 'Error: not a valid dataset name'
        sys.exit()
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2)

    model.cuda(gpu)

    print 'Ready to test network.'

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0
    n_margins = 20
    yaw_correct = np.zeros(n_margins)
    pitch_correct = np.zeros(n_margins)
    roll_correct = np.zeros(n_margins)

    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    l1loss = torch.nn.L1Loss(size_average=False)

    for i, (images, labels, name) in enumerate(test_loader):
        images = Variable(images).cuda(gpu)
        total += labels.size(0)
        label_yaw = labels[:,0].float()
        label_pitch = labels[:,1].float()
        label_roll = labels[:,2].float()

        pre_yaw, pre_pitch, pre_roll, angles = model(images)
        yaw = angles[args.iter_ref-1][:,0].cpu().data
        pitch = angles[args.iter_ref-1][:,1].cpu().data
        roll = angles[args.iter_ref-1][:,2].cpu().data

        # Mean absolute error
        yaw_error += torch.sum(torch.abs(yaw - label_yaw) * 3)
        pitch_error += torch.sum(torch.abs(pitch - label_pitch) * 3)
        roll_error += torch.sum(torch.abs(roll - label_roll) * 3)

        # Save images with pose cube.
        # TODO: fix for larger batch size
        if args.save_viz:
            name = name[0]
            cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
            utils.plot_pose_cube(cv2_img, yaw[0] * 3 - 99, pitch[0] * 3 - 99, roll[0] * 3 - 99)
            cv2.imwrite(os.path.join('output/images', name + '.jpg'), cv2_img)

    print('Test error in degrees of the model on the ' + str(total) +
    ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw_error / total,
    pitch_error / total, roll_error / total))

    # Binned accuracy
    # for idx in xrange(len(yaw_correct)):
    #     print yaw_correct[idx] / total, pitch_correct[idx] / total, roll_correct[idx] / total
