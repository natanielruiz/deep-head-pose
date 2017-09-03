import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import sys, os, argparse

import datasets, hopenet, utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--bboxes', dest='bboxes', help='Bounding box annotations of frames')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_path = args.video_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet101 with 3 outputs.
    # model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
    # ResNet50
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # ResNet18
    # model = hopenet.Hopenet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)

    print 'Loading snapshot.'
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.RandomCrop(224), transforms.ToTensor()])

    model.cuda(gpu)

    print 'Ready to test network.'

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

    txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

    bbox_file = open(args.bboxes, 'r')
    frame_num = 1

    # TODO: support for several bounding boxes
    for line in bbox_file:
        line = line.strip('\n')
        line = line.split(' ')
        det_frame_num = int(line[0])

        print frame_num

        # Stop at a certain frame number
        if frame_num > 10000:
            out.release()
            video.release()
            bbox_file.close()
            txt_out.close()
            sys.exit(0)

        # Save all frames as they are if they don't have bbox annotation.
        while frame_num < det_frame_num:
            ret, frame = video.read()
            if ret == False:
                out.release()
                video.release()
                bbox_file.close()
                txt_out.close()
                sys.exit(0)
            out.write(frame)
            frame_num += 1

        ret,frame = video.read()
        if ret == False:
            out.release()
            video.release()
            bbox_file.close()
            txt_out.close()
            sys.exit(0)

        x_min, y_min, x_max, y_max = int(line[1]), int(line[2]), int(line[3]), int(line[4])
        x_min -= 100
        x_max += 100
        y_min -= 200
        y_max += 50
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        # Crop image
        img = frame[y_min:y_max,x_min:x_max]
        img = Image.fromarray(img)

        # Transform
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(gpu)
        yaw, pitch, roll = model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        # Print new frame with cube and TODO: axis
        txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
        utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = 200)
        # Plot expanded bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 3)
        out.write(frame)

        frame_num += 1

    while True:
        ret, frame = video.read()
        if ret == False:
            out.release()
            video.release()
            bbox_file.close()
            txt_out.close()
            sys.exit(0)
        out.write(frame)
        frame_num += 1
