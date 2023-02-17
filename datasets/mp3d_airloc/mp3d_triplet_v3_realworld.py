#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from typing import Optional, Union
import cv2
import os
import random
import numbers
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import sys
from tqdm import tqdm
sys.path.append('.')
import pickle
import random


class mp3d(Dataset):
    def __init__(self,
                 base_dir: str,
                 datasets: list = None,
                 seqlen: int = 4,
                 dilation: Optional[int] = None,
                 stride: Optional[int] = None,
                 height: int = 480,
                 width: int = 640,
                 *,
                 return_img: bool = False,
                 return_seg: bool = True,
                 return_depth: bool = False,
                 return_points: bool = True,
                 train: bool = True,
                 group: int = 2,
                 test_scenes: list
                 ):
        self.base_dir = base_dir
        self.datasets = datasets
        self.seqlen = seqlen
        self.dilation = dilation
        self.height = height
        self.width = width
        self.return_img = return_img
        self.return_seg = return_seg
        self.return_depth = return_depth
        self.return_points = return_points
        self.is_train = train
        self.group = group
        self.test_scenes = test_scenes

        self.images = []
        self.rgb_data = []
        self.seg_data = []
        self.depth_data = []
        self.points_data = []
        self.room = []


        #Used when we just have a single image.pkl file and base_dir is the filename    
        # with open(self.base_dir,'rb') as f:
        #     images = pickle.load(f)
        # self.images = images["images"]
        # self.num_images = len(self.images)

        #Used when we just have multiple image.pkl files, one for each scence and base_dir is the directory    
        # test_scenes = ['8WUmhLawc2A']
        test_scenes = self.test_scenes
        if self.is_train:
            for  scene in os.listdir(base_dir):
                if scene.replace(".pkl","") not in test_scenes:
                    with open(os.path.join(self.base_dir,scene),'rb') as f:
                        image = pickle.load(f)
                    self.images += image["images"]
        else:
            for scene in os.listdir(base_dir):
                if scene.replace(".pkl","") in test_scenes:
                    with open(os.path.join(self.base_dir,scene),'rb') as f:
                        image = pickle.load(f)
                    self.images += image["images"]
        self.num_images = len(self.images)

        self.room = [self.images[i]["room_image_name"][0] for i in range(self.num_images)]
        self.points = [self.images[i]["points"] for i in range(self.num_images)]
        self.descs = [self.images[i]["descs"] for i in range(self.num_images)]
        self.ids = [self.images[i]["ids"] for i in range(self.num_images)]
                    

    def __len__(self):
        return int(self.num_images)

    def __getitem__(self, idx):
        """Code to return the data in Image format Images"""  
        
        # positive_idxs = np.where(np.array(self.room) == self.room[idx])[0]
        # positive_idxs = np.delete(positive_idxs, np.argwhere(positive_idxs==idx)).tolist()  
        # negative_idxs = np.where(np.array(self.room) != self.room[idx])[0].tolist()

        anchor_idx = range(idx,idx+1)
        # print(anchor_idx)
        # anchor_idx.append(idx)
        anchor_mapping = map(self.images.__getitem__, anchor_idx)
        anchor = list(anchor_mapping)
        
        if self.is_train:
                  
             #Removing self from id
            positive_idx = random.sample(positive_idxs,self.group)
            positive_mapping = map(self.images.__getitem__, positive_idx)
            positive = list(positive_mapping)

            negative_idx = random.sample(negative_idxs,self.group)
            negative_mapping = map(self.images.__getitem__, negative_idx)
            negative = list(negative_mapping)
            
            return anchor,positive,negative
        
        else:
            return [anchor]

