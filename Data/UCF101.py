import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import pickle
import os
from .spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

from .NTUARD_Dataset_Train import NTUARD_TRAIN
from .contrastive_dataset_NTU import ContrastiveDataset, ContrastiveSingleClip
import pdb
from .transforms import AugmentVideo, ConsistentRandomCrop

import torch
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_ucf101(root='Data', frames_path=''):
    from UCF101_Dataset_Train import UCF101TRAIN

    from UCF101_Dataset_Test import UCF101TEST
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


def get_ntuard(root='Data', frames_path='/datasets/NTU-ARD/frames-240x135', num_clips=1, cross_subject=False, contrastive=False, augment=True, hard_positive=False, random_temporal=True, multiview=False, args=None, num_frames=8, normal_mean=128., normal_std=128):
    ## augmentations
    crop_scales = [1.0]
    for _ in range(1, 5):
        crop_scales.append(crop_scales[-1] * 0.84089641525)  ##smallest scale is 0.5
    transform_consistent = Compose([
        AugmentVideo(mean=normal_mean, std=normal_std, tensor_function=ToTensor(1))
    ])

    transform_train = Compose([
        ConsistentRandomCrop(norm=True, mean=normal_mean, std=normal_std, tensor_function=ToTensor(1))
    ])

    transform_val = Compose([
        ConsistentRandomCrop(norm=True, mean=normal_mean, std=normal_std, tensor_function=ToTensor(1))
    ])
    '''
    transform_contrastive = Compose([
        Scale(136),
        CenterCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], 0.8),
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], 0.2),
        transforms.GaussianBlur(112 // 10),
        ToTensor(1),
    ])
    '''
    keys = dir(args)

    train_datasets = []
    transform_contrastive = transform_train
    if 'temporally_consistent_spatial_augment' in keys and args.temporally_consistent_spatial_augment:

        transform_contrastive = [transform_contrastive, transform_consistent]
        transform_train = transform_val = transform_consistent

    train_dataset = NTUARD_TRAIN(root=root, train=True, fold=1, cross_subject=cross_subject, num_frames=num_frames,
                                transform=transform_train,
                                num_clips=num_clips, frames_path=frames_path, args=args, pseudo_label=False)
    test_dataset = NTUARD_TRAIN(root=root, train=False, fold=1, cross_subject=cross_subject, num_frames=num_frames,
                                transform=transform_val, num_clips=num_clips, frames_path=frames_path,
                                args=args if 'pseudo_label' in keys else None)

    if 'contrastive_single_clip' in keys and args.contrastive_single_clip:
        # uses only transform_consistent
        contrastive_dataset = ContrastiveSingleClip(root=root, fold=1, transform=transform_consistent, num_clips=2,
                                             frames_path=frames_path, cross_subject=cross_subject, num_frames=num_frames,
                                             hard_positive=False, random_temporal=True,
                                             multiview=False, args=args)
        return [contrastive_dataset], test_dataset

    contrastive_dataset = ContrastiveDataset(root=root, fold=1, transform=transform_contrastive, num_clips=num_clips,
                                             frames_path=frames_path, cross_subject=cross_subject, num_frames=num_frames,
                                             hard_positive=hard_positive, random_temporal=random_temporal,
                                             multiview=multiview, args=args)
    if contrastive:
        train_datasets.append(contrastive_dataset)
    elif 'combined_multiview_training' in keys and args.combined_multiview_training:
        train_datasets.append(contrastive_dataset)
        train_datasets.append(train_dataset)
    else:
        train_datasets.append(train_dataset)
        if 'pseudo_label' in keys and args.pseudo_label:
            pseudo_label_dataset = NTUARD_TRAIN(root=root, train=True, fold=1, cross_subject=cross_subject,
                                         transform=transform_train,
                                         num_clips=num_clips, frames_path=frames_path,
                                         args=args, pseudo_label=True)
            train_datasets.append(pseudo_label_dataset)

    if 'semi_supervised_contrastive_joint_training' in keys and args.semi_supervised_contrastive_joint_training:
        train_datasets.append(train_dataset)

    return train_datasets, test_dataset


def get_multitask_dere_dataset(root='Data', frames_path='/datasets/NTU-ARD/frames-240x135', num_clips=1, cross_subject=False, multiview=False, args=None, no_frames=8, normal_mean=128., normal_std=128):
    transform_train = Compose([
        Scale(136),
        CenterCrop(112),
        ToTensor(1),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    transform_val = Compose([
        Scale(136),
        CenterCrop(112),
        ToTensor(1),
        transforms.Normalize(mean=normal_mean, std=normal_std)

    ])

    train_datasets = []

    train_datasets.append(ContrastiveDataset(root=root, fold=1, transform=transform_train, num_clips=num_clips,
                                             frames_path=frames_path, cross_subject=cross_subject, num_frames=no_frames,
                                             hard_positive=False, random_temporal=False,
                                             multiview=multiview, args=args))
    train_datasets.append(ContrastiveDataset(root=root, fold=1, transform=transform_train, num_clips=num_clips,
                                             frames_path=frames_path, cross_subject=cross_subject, num_frames=no_frames,
                                             hard_positive=False, random_temporal=False,
                                             multiview=multiview, args=args, unlabel=True))
    train_datasets[1].targets = [args.num_class+1 for _ in train_datasets[1].targets]
    test_dataset = NTUARD_TRAIN(root=root, train=False, fold=1, cross_subject=cross_subject, num_frames=no_frames, transform=transform_val, num_clips=num_clips, frames_path=frames_path, args=None)


    return train_datasets, test_dataset


if __name__ == "__main__":
    print("test")
    import time
    start = time.time()
    try:
        class arguments:
            def __init__(self):
                self.tcl = False
                self.pseudo_label = False
                self.percentage = 0.01
                self.num_class = 60
        print(os.getcwd())
        train_dataset, test_dataset=get_ntuard(root='.', contrastive=False, args=arguments())
        from torch.utils.data import random_split
        from collections import Counter
        import numpy
        percentage = 0.01
        train_dataset = train_dataset
        split_a, split_b = random_split(train_dataset, (round(percentage * len(train_dataset)), round((1 - percentage) * len(train_dataset))))
        #split_a = train_dataset
        print(len(split_a))
        print(len(train_dataset))
        #print(len([i for i in split_a.targets if i != 61]))
        c = Counter([split_a.dataset.targets[i] for i in split_a.indices])
        #c = Counter([i for i in split_a.targets if i != 61])
        print(np.mean(np.array(list(c.values()))))
        print(np.var(np.array(list(c.values()))))
        '''counts={}
        for i in [split_a.dataset.targets[i] for i in split_a.indices]:
        print()'''
    except Exception as e:
        print(e)
    print("test")
    print("after: ", time.time()-start)