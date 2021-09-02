import os
import numpy as np
height=135
width=240
crop_size=112
center_crop=True
num_frames=8
crop_pos_x = np.random.randint(0, height - crop_size)
crop_pos_y = np.random.randint(0, width - crop_size)
if center_crop:
    # if we need to crop only from the center of the frame
    crop_pos_y = np.random.randint(50, width - crop_size - 50)

print(crop_pos_x)
print(crop_pos_y)
'''
import random
import os
subsets = []
percents = [0.01, 0.05, 0.10, 0.20, 0.40, 0.6, 0.8, 1]
percents.reverse()
build_paths=[]
#read in dataset
base_file_name="cross_view_split"
sub_dir = f'{base_file_name}_subsets'
os.makedirs('ntuTrainTestList/'+sub_dir, exist_ok=True)
with open(f'ntuTrainTestList/{base_file_name}.list') as f:
    for line in f:
        if line != "" and 'S' in line:
            build_paths.append(line)
for percent in percents:
    print(percent)

    if percent == 1:
        subsets.append(build_paths)
        continue
    with open(f'ntuTrainTestList/{sub_dir}/{percent}.list', 'r') as f:
        indicies = []
        for item in f:
            indicies.append(item)
        subsets.append(indicies)


    indices = list(range(len(subsets[-1])))
    sampled_indices = random.sample(indices, int(len(build_paths) * (percent)))
    subsets.append([subsets[-1][i] for i in sampled_indices])
    with open(f'ntuTrainTestList/{sub_dir}/{percent}.list', 'w') as f:
        for item in subsets[-1]:
            f.write(item)
            
with open(_get_create_subset_exist(root=".", p=0.65), 'r') as f:
    print(0.65, len([line for line in f]), len(subsets[0])*0.65)
for i in subsets:
    print(len(i))
'''


def _get_create_subset_exist(root, p):
    # get all splits available
    files = os.listdir(f'{root}/ntuTrainTestList/cross_view_split_subsets')
    percentages = [float(f_name.split('.list')[0]) for f_name in files]
    path_name = f"{root}/ntuTrainTestList/cross_view_split_subsets/{p}.list"
    with open(os.path.join(root, 'ntuTrainTestList', 'cross_view_split.list'), 'r') as f:
        total_num_files = len([line for line in f])
    if p not in percentages:
        percentages.sort()
        for i, p_existing_split in enumerate(percentages):
            if p < p_existing_split:
                k = p if i == 0 else p - percentages[i - 1]  # percentage to sample from larger subset
                k = k * total_num_files  # number of files to sample
                with open(f"{root}/ntuTrainTestList/cross_view_split_subsets/{p_existing_split}.list", 'r') as f:
                    unique_files_larger_set = set([line for line in f])
                if i == 0:
                    # smaller than the smallest
                    unique_smaller_subset = set()
                else:
                    with open(f"{root}/ntuTrainTestList/cross_view_split_subsets/{percentages[i - 1]}.list",
                              'r') as f:
                        unique_smaller_subset = set([line for line in f])
                selected_samples = random.sample(unique_files_larger_set - unique_smaller_subset, int(k))
                with open(path_name, 'w') as f:
                    for item in selected_samples:
                        f.write(item)
                    for item in unique_smaller_subset:
                        f.write(item)
    return path_name



'''
import torch
from transforms import AugmentVideo
import os
from PIL import Image
import cv2
from torchvision import transforms

clips = []
video_container = []
from torchvision import transforms
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)


transform_train = Compose([
    Scale(136),
    CenterCrop(112),
    ToTensor()])

transform_contrastive = Compose([
        Scale(136),
        CenterCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], 0.8),
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], 0.2),
        transforms.GaussianBlur(112 // 10),
        ToTensor()])
transform_consistent = Compose([
    torch.stack,
    AugmentVideo(0.5, 0.5)
])
for item in range(30,46, 2):
    image_name = '{:03d}.jpg'.format(item)
    image_path = os.path.join('../sample_video', image_name)
    current_image = Image.open(image_path).convert('RGB')
    current_image = transforms.ToTensor()(current_image) #TODO key add
    #print(current_image.shape)
    # TODO removable
    #current_image = transform_contrastive(current_image)
    video_container.append(current_image)

clip = transform_consistent(video_container) #TODO key add
#clip = video_container #TODO remove
clip = torch.stack(clip, 0).permute(0, 2, 3,1)
clips.append(clip)
clips=torch.stack(clips)

for i in range(8):
    #RGB_img = cv2.cvtColor(positives_viewA[i], cv2.COLOR_BGR2RGB)
    RGB_img = clips[0][i].numpy()
    print(RGB_img.shape)
    #RGB_img2 = cv2.cvtColor(positives_viewB[i], cv2.COLOR_BGR2RGB)
    cv2.imshow('view A', RGB_img)
    #cv2.imshow('view B', RGB_img2)
    #cv2.moveWindow('view B', 200, 0)
    cv2.waitKey(0)
'''