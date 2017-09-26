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

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--num_epochs_ft', dest='num_epochs_ft', help='Maximum number of finetuning epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.001, type=float)
    parser.add_argument('--iter_ref', dest='iter_ref', help='Number of iterative refinement passes.',
          default=1, type=int)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = []
    b.append(model.conv1)
    b.append(model.bn1)
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = []
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    b = []
    b.append(model.fc_yaw)
    b.append(model.fc_pitch)
    b.append(model.fc_roll)
    b.append(model.fc_finetune)
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(snapshot)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    num_epochs_ft = args.num_epochs_ft
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # ResNet101 with 3 outputs
    # model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
    # ResNet50
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66, args.iter_ref)
    # ResNet18
    # model = hopenet.Hopenet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
    load_filtered_state_dict(model, model_zoo.load_url(model_urls['resnet50']))

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    else:
        print 'Error: not a valid dataset name'
        sys.exit()
    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    model.cuda(gpu)
    softmax = nn.Softmax()
    criterion = nn.CrossEntropyLoss().cuda()
    reg_criterion = nn.MSELoss().cuda()
    smooth_l1_loss = nn.SmoothL1Loss().cuda()
    # Regression loss coefficient
    alpha = args.alpha

    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': args.lr},
                                  {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                   lr = args.lr)

    print 'Ready to train network.'

    print 'First phase of training.'
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images.cuda(gpu))
            label_yaw = Variable(labels[:,0].cuda(gpu))
            label_pitch = Variable(labels[:,1].cuda(gpu))
            label_roll = Variable(labels[:,2].cuda(gpu))

            label_angles = Variable(cont_labels[:,:3].cuda(gpu))
            label_yaw_cont = Variable(cont_labels[:,0].cuda(gpu))
            label_pitch_cont = Variable(cont_labels[:,1].cuda(gpu))
            label_roll_cont = Variable(cont_labels[:,2].cuda(gpu))

            optimizer.zero_grad()
            model.zero_grad()

            pre_yaw, pre_pitch, pre_roll, angles = model(images)

            # Cross entropy loss
            loss_yaw = criterion(pre_yaw, label_yaw)
            loss_pitch = criterion(pre_pitch, label_pitch)
            loss_roll = criterion(pre_roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(pre_yaw)
            pitch_predicted = softmax(pre_pitch)
            roll_predicted = softmax(pre_roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.Tensor(1).cuda(gpu) for _ in range(len(loss_seq))]
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                       %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size, loss_yaw.data[0], loss_pitch.data[0], loss_roll.data[0]))
                # if epoch == 0:
                #     torch.save(model.state_dict(),
                #     'output/snapshots/' + args.output_string + '_iter_'+ str(i+1) + '.pkl')

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print 'Taking snapshot...'
            torch.save(model.state_dict(),
            'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')

    print 'Second phase of training (finetuning layer).'
    for epoch in range(num_epochs_ft):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images.cuda(gpu))
            label_yaw = Variable(labels[:,0].cuda(gpu))
            label_pitch = Variable(labels[:,1].cuda(gpu))
            label_roll = Variable(labels[:,2].cuda(gpu))

            label_angles = Variable(cont_labels[:,:3].cuda(gpu))
            label_yaw_cont = Variable(cont_labels[:,0].cuda(gpu))
            label_pitch_cont = Variable(cont_labels[:,1].cuda(gpu))
            label_roll_cont = Variable(cont_labels[:,2].cuda(gpu))

            optimizer.zero_grad()
            model.zero_grad()

            pre_yaw, pre_pitch, pre_roll, angles = model(images)

            # Cross entropy loss
            loss_yaw = criterion(pre_yaw, label_yaw)
            loss_pitch = criterion(pre_pitch, label_pitch)
            loss_roll = criterion(pre_roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(pre_yaw)
            pitch_predicted = softmax(pre_pitch)
            roll_predicted = softmax(pre_roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            # Finetuning loss
            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            for idx in xrange(1,len(angles)):
                label_angles_residuals = label_angles - angles[0] * 3 - 99
                for idy in xrange(1,idx):
                    label_angles_residuals += angles[idy] * 3 - 99
                label_angles_residuals = label_angles_residuals.detach()
                # Reconvert to other unit
                label_angles_residuals = label_angles_residuals / 3.0 + 33
                loss_angles = smooth_l1_loss(angles[idx], label_angles_residuals)
                loss_seq.append(loss_angles)

            grad_seq = [torch.Tensor(1).cuda(gpu) for _ in range(len(loss_seq))]
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: pre-yaw %.4f, pre-pitch %.4f, pre-roll %.4f, finetuning %.4f'
                       %(epoch+1, num_epochs_ft, i+1, len(pose_dataset)//batch_size, loss_yaw.data[0], loss_pitch.data[0], loss_roll.data[0], loss_angles.data[0]))
                # if epoch == 0:
                #     torch.save(model.state_dict(),
                #     'output/snapshots/' + args.output_string + '_iter_'+ str(i+1) + '.pkl')

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs_ft:
            print 'Taking snapshot...'
            torch.save(model.state_dict(),
            'output/snapshots/' + args.output_string + '_epoch_'+ str(num_epochs+epoch+1) + '.pkl')
