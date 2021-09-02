from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import pickle, h5py
import cv2
import torch
import torch
from torch.autograd import Variable
import json
from PIL import Image
import pdb
import random
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
from torchvision import transforms
from .transforms import AugmentVideo, ConsistentRandomCrop



class NTUARD_TRAIN(Dataset):
    def __init__(self, root = '', train=True, fold=1, transform=None, frames_path='', num_clips=3, num_frames=8, cross_subject=False, pseudo_label=False, args=None):

        self.num_clips = num_clips
        self.pseudo_label=pseudo_label
        self.args = args
        self.num_frames = num_frames
        self.frames_path = frames_path
        self.root = root
        self.train = train
        self.cross_subject = cross_subject
        if not cross_subject:
            if self.train:
                self.views=[2,3]
            else:
                self.views=[1]
        else:
            self.views=[1,2,3]

        self.fold = fold
        self.video_paths, self.targets = self.build_paths()
        self.targets = np.array(self.targets)
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_dict, video_label = self.video_paths[idx], self.targets[idx]
        video = self.get_video(video_dict)
        return video, video_label-1

    def get_video(self, video_dict):
        no_frames = video_dict['no_frames']-1
        skip_rate = 1
        total_frames = self.num_frames*skip_rate

        if total_frames*self.num_clips > no_frames:
            skip_rate = skip_rate -1
            if skip_rate == 0:
                skip_rate = 1
            total_frames = (self.num_frames)*skip_rate

        try:
            #start_frame = random.randint(0, no_frames - total_frames) ## 32, 16 frames
            ids = np.sort(np.random.randint([i * no_frames // self.num_clips for i in range(self.num_clips)],
                                      [i * (no_frames // self.num_clips) - total_frames for i in range(1, self.num_clips+1)], self.num_clips))

        except:
            #print("Frame exception", no_frames, total_frames, self.num_clips*total_frames)
            #start_frame = 0
            ids = [i*(total_frames-skip_rate) for i in range(self.num_clips)]
            #print("passed exception")
        clips = []
        types_of_transforms=[type(t) for t in self.transform.transforms]
        for start_frame in ids:
            video_container = []
            for item in range(start_frame, start_frame + total_frames, skip_rate):
                image_name = '{:03d}.jpg'.format(item)
                image_path = os.path.join(video_dict['path'], image_name)
                current_image = Image.open(image_path).convert('RGB')
                video_container.append(current_image)

            if self.transform is not None:
                #self.transform.randomize_parameters()
                if AugmentVideo in types_of_transforms or ConsistentRandomCrop in types_of_transforms:
                    clip = self.transform(video_container)
                else:
                    clip = [self.transform(img) for img in video_container] #[transforms.functional.normalize(self.transform(img), normal_mean, normal_std) for img in video_container]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clips.append(clip)
        return torch.stack(clips)

    def _decrypt_vid_name(self, vid):
        scene = int(vid[1:4])
        pid = int(vid[5:8])
        rid = int(vid[9:12])
        action = int(vid[13:16])

        return scene, pid, rid, action
    def _get_create_subset(self, p):
        # get all splits available
        files = os.listdir(f'{self.root}/ntuTrainTestList/cross_view_split_subsets')
        percentages = [float(f_name.split('.list')[0]) for f_name in files]
        path_name = f"{self.root}/ntuTrainTestList/cross_view_split_subsets/{p}.list"
        with open(os.path.join(self.root, 'ntuTrainTestList', 'cross_view_split.list'), 'r') as f:
            total_num_files = len([line for line in f])
        if p not in percentages:
            percentages.sort()
            for i, p_existing_split in enumerate(percentages):
                if p < p_existing_split:
                    k = p if i == 0 else p - percentages[i-1] # percentage to sample from larger subset
                    k = k*total_num_files # number of files to sample
                    with open(f"{self.root}/ntuTrainTestList/cross_view_split_subsets/{p_existing_split}.list", 'r') as f:
                        unique_files_larger_set = set([line for line in f])
                    if i == 0:
                        #smaller than the smallest
                        unique_smaller_subset = set()
                    else:
                        with open(f"{self.root}/ntuTrainTestList/cross_view_split_subsets/{percentages[i-1]}.list", 'r') as f:
                            unique_smaller_subset = set([line for line in f])
                    selected_samples = random.sample(unique_files_larger_set-unique_smaller_subset, int(k))
                    with open(path_name, 'w') as f:
                        for item in selected_samples:
                            f.write(item)
                        for item in unique_smaller_subset:
                            f.write(item)
        return path_name



    def build_paths(self):
        data_paths = []
        targets = []
        # train and val list are the splits for cross subject
        # video and video1 are the splits for cross view
        if self.train and self.cross_subject:
            annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'train.list')
        elif self.cross_subject:
            annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'val.list')
        else:
            if self.args and self.args.percentage < 1 and not self.pseudo_label:
                # check if exists

                annotation_path = self._get_create_subset(self.args.percentage)
            else:
                annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'cross_view_split.list')

            '''if self.pseudo_label:
                subset_path_ = self._get_create_subset(self.root, self.args.percentage)
                with open(subset_path_) as f:
                    labeled_examples = [line for line in f]'''
        
        with open(annotation_path, "r") as fid:
            dataList = fid.readlines()
            for x in dataList:
                original_line = x
                x = x.split()
                video_name = x[0]
                scene, pid, rid, action = self._decrypt_vid_name(video_name.split("/")[1])
                for view in self.views:
                    if int(x[view]) > self.num_clips*self.num_frames:
                        data_paths.append({'path':os.path.join(self.frames_path, video_name, str(view)), 'no_frames':int(x[view])})
                        if self.pseudo_label and self.train:
                            action = self.args.num_class + 1 # if original_line not in labeled_examples else action
                        targets.append(action)
                    else:
                        print("Insufficient # of frames: ", video_name, view, x[view], self.num_clips*self.num_frames)
        if self.args and self.pseudo_label and self.train and False:

            indices = list(range(len(data_paths)))
            sampled_indices = random.sample(indices, int(len(indices)*(1-self.args.percentage)))
            for i in sampled_indices:
                targets[i] = self.args.num_class + 1

        return data_paths, targets

#dataset=NTUARD_TRAIN(root='',frames_path='/datasets/NTU-ARD/frames-240x135')