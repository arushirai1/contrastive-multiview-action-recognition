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
import random



class ContrastiveDataset(Dataset):
    def __init__(self, root = '', fold=1, transform=None, frames_path='', num_clips=2, num_frames=8, multiview=False, hard_positive=False, cross_subject=False, random_temporal=True, args=None, unlabel=False):
        self.unlabel = unlabel
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frames_path = frames_path
        self.root = root
        self.cross_subject = cross_subject
        self.args = args
        if self.args.tcl:
            self.skip_rates = [1,2]
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

        for i, view_dict in enumerate(video_paths):
            if self.get_flag('tcl'):
                video, ids_tmp = self.get_video(view_dict, ids, skip_rate=self.skip_rates[i], transform=self.transform)
            elif self.get_flag('temporally_consistent_spatial_augment'):
                video, ids_tmp = self.get_video(view_dict, ids, transform=self.transform[i])
            else:
                video, ids_tmp = self.get_video(view_dict, ids, transform=self.transform)
            if not self.random_temporal:
                ids = ids_tmp
            positives.append(video)
        if self.args.combined_multiview_training or self.args.joint_only_multiview_training:
            positives = torch.cat(positives, dim=0)
        if self.get_flag('classify_view'):
            return positives, [video_label-1, [int(video_path['path'][-1])-2 for video_path in video_paths]]
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

    def get_video(self, video_dict, ids=[], skip_rate=2, transform=None):
        no_frames = video_dict['no_frames']-1
        total_frames = self.num_frames*skip_rate

        if total_frames*self.num_clips > no_frames:
            skip_rate = skip_rate -1
            if skip_rate == 0:
                skip_rate = 1
            total_frames = (self.num_frames)*skip_rate
        ids = self._get_ids(no_frames, total_frames, skip_rate, ids)
        clips = []
        types_of_transforms=[type(t) for t in self.transform.transforms]
        for start_frame in ids:
            video_container = []
            for item in range(start_frame, start_frame + total_frames, skip_rate):
                image_name = '{:03d}.jpg'.format(item)
                image_path = os.path.join(video_dict['path'], image_name)
                current_image = Image.open(image_path).convert('RGB')
                video_container.append(current_image)

            if transform is not None:
                #self.transform.randomize_parameters()
                if AugmentVideo in types_of_transforms or ConsistentRandomCrop in types_of_transforms:
                    clip = transform(video_container)
                else:
                    clip = [transform(img) for img in video_container] #[transforms.functional.normalize(self.transform(img), normal_mean, normal_std) for img in video_container]
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
    def get_flag(self, name):
        keys = dir(self.args)
        if name not in keys:
            return False
        if name == 'temporally_consistent_spatial_augment' and self.args.temporally_consistent_spatial_augment:
            return True
        elif name == 'tcl' and self.args.tcl:
            return True
        elif name == 'classify_view' and self.args.classify_view:
            return True
        return False
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

    def get_annotation_path(self):
        if self.cross_subject:
            annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'train.list')
        else:
            if self.args and self.args.percentage < 1 and not self.unlabel:
                # check if exists
                annotation_path = self._get_create_subset(self.args.percentage)
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
                if self.get_flag('tcl') or self.get_flag('temporally_consistent_spatial_augment'):
                    for view in self.views:
                        positive_pair = []
                        for _ in self.views:
                            positive_pair.append(
                                {'path': os.path.join(self.frames_path, video_name, str(view)), 'no_frames': int(x[view])})
                        if self.unlabel:
                            action = self.args.num_class + 1
                        targets.append(action)
                        data_paths.append(positive_pair)
                else:
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

class ContrastiveSingleClip(ContrastiveDataset):
    def __init__(self, root = '', fold=1, transform=None, frames_path='', num_clips=2, num_frames=8, multiview=False, hard_positive=False, cross_subject=False, random_temporal=True, args=None, unlabel=False):
        super(ContrastiveSingleClip, self).__init__(root, fold, transform, frames_path, 2, num_frames, multiview, hard_positive, cross_subject, random_temporal, args, unlabel) # 2 is the number of clips
        self.video_paths, self.targets = self.build_paths()
    def __getitem__(self, idx):
        video_paths, video_label = self.video_paths[idx], self.targets[idx]
        ids = []
        video, ids_tmp = self.get_video(video_paths, ids, transform=self.transform)
        return (video[0], video[1]), video_label-1

    def build_paths(self):
        data_paths = []
        targets = []
        # train and val list are the splits for cross subject
        # video and video1 are the splits for cross view
        if self.cross_subject:
            annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'train.list')
        else:
            if self.args and self.args.percentage < 1:
                # check if exists

                annotation_path = self._get_create_subset(self.args.percentage)
            else:
                annotation_path = os.path.join(self.root, 'ntuTrainTestList', 'cross_view_split.list')

        with open(annotation_path, "r") as fid:
            dataList = fid.readlines()
            for x in dataList:
                original_line = x
                x = x.split()
                video_name = x[0]
                scene, pid, rid, action = self._decrypt_vid_name(video_name.split("/")[1])
                for view in self.views:
                    if int(x[view]) > self.num_clips * self.num_frames:
                        data_paths.append(
                            {'path': os.path.join(self.frames_path, video_name, str(view)), 'no_frames': int(x[view])})
                        targets.append(action)
                    else:
                        print("Insufficient # of frames: ", video_name, view, x[view], self.num_clips * self.num_frames)

        return data_paths, targets