from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
# from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import SubsetRandomSampler, Dataset
from torch import nn, optim
from torch.autograd import Variable
import torchvision

class ImageFolderWithRatings(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ratings):
        super().__init__(root, transform)
        self.ratings = ratings
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithRatings, self).__getitem__(index)
        path = self.imgs[index][0]
        fname = path.split("/")[-1]
        fname = fname.split(".")[0]
        rating = self.ratings[np.where(self.ratings[:, 0] == fname)][0][1]
        tuple_with_rating= (original_tuple + (rating,))
        return tuple_with_rating

class ImageFolderWithRatingsAndFilenames(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ratings):
        super().__init__(root, transform)
        self.ratings = ratings
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithRatingsAndFilenames, self).__getitem__(index)
        path = self.imgs[index][0]
        fname = path.split("/")[-1]
        imgname = fname.split(".")[0]
        rating = self.ratings[np.where(self.ratings[:, 0] == imgname)][0][1]
        new_tuple = (original_tuple + (rating,fname,))
        return new_tuple

class CustomDataLoader(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(self.all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        img_name = self.total_imgs[idx]
        path_plus_image = img_name, tensor_image
        return path_plus_image


def get_subset(indices, start, end):
    return indices[start : start + end]
