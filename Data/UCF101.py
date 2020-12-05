import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import pickle
import os
from .spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

from .UCF101_Dataset_Train import UCF101TRAIN
from .UCF101_Dataset_Test import UCF101TEST
import torch

normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_ucf101(root='Data', frames_path=''):
    ## augmentations
    crop_scales = [1.0]
    for _ in range(1, 5):
        crop_scales.append(crop_scales[-1] * 0.84089641525) ##smallest scale is 0.5

    transform_train = Compose([
            Scale(136),
            MultiScaleRandomCrop(crop_scales, 112),
            RandomHorizontalFlip(),
            ToTensor(1),
        ])
    
    transform_val = transforms.Compose([
            Scale(136),
            CenterCrop(112),
            ToTensor(1),
            transforms.Normalize(mean=normal_mean, std=normal_std)
        ])

    train_dataset = UCF101TRAIN (root=root, train=True, fold=1, transform=transform_train, frames_path=frames_path)
    test_dataset = UCF101TEST(root=root, train=False, fold=1, transform=transform_val, frames_path=frames_path)

    return train_dataset, test_dataset


if __name__ == "__main__":
    get_ucf101()
