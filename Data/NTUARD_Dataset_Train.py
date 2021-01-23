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



class NTUARD_TRAIN(Dataset):
    def __init__(self, root = '', train=True, fold=1, transform=None, frames_path='', num_clips=3, num_frames=8, cross_subject=False):

        self.num_clips = num_clips
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
        for start_frame in ids:
            video_container = []
            for item in range(start_frame, start_frame + total_frames, skip_rate):
                image_name = '{:03d}.jpg'.format(item)
                image_path = os.path.join(video_dict['path'], image_name)
                current_image = Image.open(image_path).convert('RGB')
                video_container.append(current_image)

            if self.transform is not None:
                self.transform.randomize_parameters()
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
            annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'cross_view_split.list')
        
        with open(annotation_path, "r") as fid:
            dataList = fid.readlines()
            for x in dataList:
                x = x.split()
                video_name = x[0]
                scene, pid, rid, action = self._decrypt_vid_name(video_name.split("/")[1])
                for view in self.views:
                    data_paths.append({'path':os.path.join(self.frames_path, video_name, str(view)), 'no_frames':int(x[view])})
                    targets.append(action)
        return data_paths, targets

#dataset=NTUARD_TRAIN(root='',frames_path='/datasets/NTU-ARD/frames-240x135')