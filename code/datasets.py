import numpy as np
import torch
import cv2
from torch.utils.data.dataset import Dataset
import os
from PIL import Image

import utils

class Pose_300W_LP(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert('RGB')

        pose = utils.get_ypr_from_mat(os.path.join(self.data_dir, self.y_train[index] + self.annot_ext))
        label = torch.FloatTensor(pose)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert('RGB')

        pose = utils.get_ypr_from_mat(os.path.join(self.data_dir, self.y_train[index] + self.annot_ext))
        label = torch.FloatTensor(pose)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length

class Pose_300W_LP_binned(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert('RGB')

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(os.path.join(self.data_dir, self.y_train[index] + self.annot_ext))
        # And convert to positive degrees.
        pose = pose * 180 / np.pi + 90

        label = torch.FloatTensor(pose)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines
