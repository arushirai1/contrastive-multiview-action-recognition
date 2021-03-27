import os
import random
from Data.contrastive_dataset_NTU import ContrastiveDataset
from Data.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from torchvision import transforms
import pickle

# PARAMS
out_dir = 'samples'
remote = False
root='Data'
frames_path='/datasets/NTU-ARD/frames-240x135'
num_clips=1
cross_subject=False
contrastive=True
augment=False
hard_positive=False
random_temporal=False

if remote:
    transform_train = Compose([
        CenterCrop(112),
        ToTensor(1),
    ])

    transform_val = Compose([
        CenterCrop(112),
        ToTensor(1),
    ])
    transform_contrastive = Compose([
        Scale(136),
        CenterCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], 0.8),
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], 0.2),
        transforms.GaussianBlur(112 // 10),
        ToTensor(1),
    ])

    if contrastive:
        if not augment:
            transform_contrastive = transform_train
    os.makedirs(out_dir, exist_ok=True)
    contrastive_dataset = ContrastiveDataset(root=root, fold=1, transform=transform_contrastive, num_clips=num_clips,
                                             frames_path=frames_path, cross_subject=cross_subject,
                                             hard_positive=hard_positive, random_temporal=random_temporal)
    samples = random.sample(list(range(0, len(contrastive_dataset))), k=5)
    for sample in samples:
        positives, _ = contrastive_dataset.__getitem__(sample)

        with open(f'{out_dir}/{sample}.pickle', 'wb') as f:
            pickle.dump(positives, f)
else:
    ## read in samples
    import numpy as np, cv2
    for sample in os.listdir(out_dir):
        with open(f'{out_dir}/{sample}', 'rb') as f:
            positives = pickle.load(f)
            positives_viewA = np.transpose(positives[0].squeeze(0).numpy(), (1, 2, 3, 0)).astype(np.uint8)
            positives_viewB = np.transpose(positives[1].squeeze(0).numpy(), (1, 2, 3, 0)).astype(np.uint8)

            for i in range(8):
                RGB_img = cv2.cvtColor(positives_viewA[i], cv2.COLOR_BGR2RGB)
                RGB_img2 = cv2.cvtColor(positives_viewB[i], cv2.COLOR_BGR2RGB)
                cv2.imshow('view A', RGB_img)
                cv2.imshow('view B', RGB_img2)
                cv2.moveWindow('view B', 200, 0)
                cv2.waitKey(0)


