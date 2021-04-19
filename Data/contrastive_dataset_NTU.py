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

import random



class ContrastiveDataset(Dataset):
    def __init__(self, root = '', fold=1, transform=None, frames_path='', num_clips=2, num_frames=8, multiview=False, hard_positive=False, cross_subject=False, random_temporal=True, args=None):

        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frames_path = frames_path
        self.root = root
        self.cross_subject = cross_subject
        self.args = args
        if not cross_subject:
            self.views=[2,3]
        else:
            self.views=[1,2,3]

        self.fold = fold
        if hard_positive:
            self.video_paths, self.targets = self.build_hard_positive_paths(multiview)
        else:
            self.video_paths, self.targets = self.build_paths()
        self.targets = np.array(self.targets)
        self.transform = transform
        self.random_temporal = random_temporal

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_paths, video_label = self.video_paths[idx], self.targets[idx]
        positives = []
        ids = []
        for view_dict in video_paths:
                video, ids_tmp = self.get_video(view_dict, ids)
                if not self.random_temporal:
                    ids = ids_tmp
                positives.append(video)
        if self.args.combined_multiview_training or self.args.joint_only_multiview_training:
            positives = torch.cat(positives, dim=0)
        return positives, video_label-1
    def _get_ids(self, no_frames, total_frames, skip_rate, ids):
        def handle_edge_case(ratio):
            potential = abs(int(ratio*no_frames))
            if potential+total_frames >= no_frames:
                return no_frames-total_frames
            return potential
        if len(ids) == 0:
            try:
                #start_frame = random.randint(0, no_frames - total_frames) ## 32, 16 frames
                ids = np.sort(np.random.randint([i * no_frames // self.num_clips for i in range(self.num_clips)],
                                          [i * (no_frames // self.num_clips) - total_frames for i in range(1, self.num_clips+1)], self.num_clips))

            except:
                #print("Frame exception", no_frames, total_frames, self.num_clips*total_frames)
                #start_frame = 0
                ids = [i*(total_frames-skip_rate) for i in range(self.num_clips)]
        else:
            ids = [handle_edge_case(ratio) for ratio in ids]
        return list(ids)

    def get_video(self, video_dict, ids=[]):
        no_frames = video_dict['no_frames']-1
        skip_rate = 2
        total_frames = self.num_frames*skip_rate

        if total_frames*self.num_clips > no_frames:
            skip_rate = skip_rate -1
            if skip_rate == 0:
                skip_rate = 1
            total_frames = (self.num_frames)*skip_rate
        ids = self._get_ids(no_frames, total_frames, skip_rate, ids)
        clips = []
        for start_frame in ids:
            video_container = []
            for item in range(start_frame, start_frame + total_frames, skip_rate):
                image_name = '{:03d}.jpg'.format(item)
                image_path = os.path.join(video_dict['path'], image_name)
                current_image = Image.open(image_path).convert('RGB')
                video_container.append(current_image)

            if self.transform is not None:
                #self.transform.randomize_parameters()
                clip = [self.transform(img) for img in video_container] #[transforms.functional.normalize(self.transform(img), normal_mean, normal_std) for img in video_container]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clips.append(clip)
        return torch.stack(clips), [i/no_frames for i in ids]

    def _decrypt_vid_name(self, vid):
        scene = int(vid[1:4])
        pid = int(vid[5:8])
        rid = int(vid[9:12])
        action = int(vid[13:16])

        return scene, pid, rid, action
    def get_action_path_dict(self, dataList):
        action_path_dict = {}
        for x in dataList:
            x = x.split()
            video_name = x[0]
            _, _, _, action = self._decrypt_vid_name(video_name.split("/")[1])
            temp_list = action_path_dict.get(action,[])
            temp_list.append(x)
            action_path_dict[action] = temp_list
        return action_path_dict

    def get_positive_pairs(self, data, pairs=[]):
        if len(data) == 1:
            # one item will be paired with two
            rand_index = random.randint(0, len(pairs)-1)
            pairs.append((pairs[rand_index][0], data[0]))
            return pairs
        elif len(data) == 0:
            return pairs
        else:
            rand_index = random.randint(0, len(data)-1)
            pairs.append((data.pop(rand_index), data.pop(0)))
            return self.get_positive_pairs(data, pairs)

    def get_pairs(self, action_path_dict):
        pairs = []
        for key in action_path_dict.keys():
            pairs_batch = self.get_positive_pairs(action_path_dict[key], [])
            pairs.extend(pairs_batch)
        return pairs

    def get_annotation_path(self):
        if self.cross_subject:
            annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'train.list')
        else:
            annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'cross_view_split.list')
        return annotation_path

    def build_paths(self):
        data_paths = []
        targets = []
        annotation_path = self.get_annotation_path()

        with open(annotation_path, "r") as fid:
            dataList = fid.readlines()
            for x in dataList:
                x = x.split()
                video_name = x[0]
                scene, pid, rid, action = self._decrypt_vid_name(video_name.split("/")[1])
                positive_pair=[]
                for view in self.views:
                    positive_pair.append({'path':os.path.join(self.frames_path, video_name, str(view)), 'no_frames':int(x[view])})
                targets.append(action)
                data_paths.append(positive_pair)
        return data_paths, targets

    def build_hard_positive_paths(self, multiview=False):
        data_paths = []
        targets = []
        annotation_path = self.get_annotation_path()
        with open(annotation_path, "r") as fid:
            dataList = fid.readlines()
            action_path_dict = self.get_action_path_dict(dataList)
            data_pairs = self.get_pairs(action_path_dict)
            if multiview:
                for pair in data_pairs:
                    _, _, _, action = self._decrypt_vid_name(pair[0][0].split("/")[1])
                    permuted_pair = [(pair[0], pair[1]), (pair[1], pair[0])]  # for only two views, TODO: Generalize
                    for pair in permuted_pair:
                        positive_pair = []
                        for i, view in enumerate(self.views):
                            video_name = pair[i][0]
                            positive_pair.append({'path': os.path.join(self.frames_path, video_name, str(view)),
                                                   'no_frames': int(pair[i][view])})
                        targets.append(action)
                        data_paths.append(positive_pair)
            else:
                for pair in data_pairs:
                    _,_,_, action = self._decrypt_vid_name(pair[0][0].split("/")[1])
                    for view in self.views:
                        positive_pair = []
                        for video_item in pair:
                            video_name = video_item[0]
                            positive_pair.append({'path':os.path.join(self.frames_path, video_name, str(view)), 'no_frames':int(video_item[view])})
                        targets.append(action)
                        data_paths.append(positive_pair)
        return data_paths, targets
