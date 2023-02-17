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
import json
import argparse
import yaml
from kornia.geometry.depth import depth_to_3d
import random

def ids_from_folder(data_path):
    #Given a folder this function extracts the ids of diferent images present in the folder
    query_list=os.listdir(data_path)
    qry_ids=[]
    for i in range(len(query_list)):
        qry_ids.append(int(query_list[i].split('_')[0]))
    qry_ids=np.array(qry_ids)
    qry_ids=np.unique(qry_ids)
    return qry_ids


def img_to_mask(img):
    obj_ids = np.unique((img))
    ann_mask = []
    for i,id in enumerate(obj_ids):
        mask = {}
        mask["mask"] = torch.from_numpy(np.where(img==id,1,0)).type(torch.float16)
        mask["id"] = id
        ann_mask.append(mask)
        
    return ann_mask


def remove_image(room_path,id):
    print("Image Removel function not implemented yet!!")


def is_good(depth_img,sem_img,configs):
    occlusion_ratio = configs["good_image"]["occlusion_ratio"]
    min_objects = configs["good_image"]["min_objects"]
    
    count = np.count_nonzero(depth_img == 0)
    ratio = count/(depth_img.shape[0]*depth_img.shape[1])

    object_count = len(np.unique(sem_img))

    return ratio < occlusion_ratio and object_count > min_objects


def get_raw_data(raw_data_folder,points_dir,id,room,configs):
    filter_images = configs["filter_images"]
    cam_config = configs["camera"]
    fx = cam_config["fx"]
    fy = cam_config["fy"]
    cx = cam_config["cx"]
    cy = cam_config["cy"]

    K = torch.tensor([
        [cy, 0., cx],
        [0., fy, cy],
        [0., 0.,  1]]).unsqueeze(0)

    #Getting Image paths
    rgb_data = os.path.join(raw_data_folder,(str(id)+"_rgb.png"))
    seg_data = os.path.join(raw_data_folder,(str(id)+"_instance-seg.png"))
    depth_data = os.path.join(raw_data_folder,(str(id)+"_depth.png"))
    points_data = os.path.join(points_dir,(str(id)+".pkl"))

    #Reading Images. Semantic and points for this case
    image = []                        
    seg = cv2.imread(seg_data,cv2.IMREAD_ANYDEPTH)
    mask = img_to_mask(seg)
    image.append(mask)
    
    with open(points_data,'rb') as fp:
        points = pickle.load(fp)
    image.append(points)

    depth = cv2.imread(depth_data,cv2.IMREAD_ANYDEPTH)
    depth_torch = torch.from_numpy(depth.astype(np.float64))
    relative_distance = depth_to_3d(depth_torch[None,None,:,:],K)
    image.append(relative_distance.squeeze())
        
    image.append([room,rgb_data])

    if filter_images and not is_good(depth,seg,configs):
        return 0

    return image


def get_image_objects(image):
    #Generating Object Points from it 
    ann_masks, points, relative_distance, roomname = image[0], image[1], image[2] ,image[3]
    
    keypoints = points['points']
    descriptors = points['point_descs']

    
    image_objects = {}
    image_objects['points'] = []
    image_objects['descs'] = []
    image_objects['relative_distance'] = []
    image_objects['ids'] = [] 
    image_objects['room_image_name'] = roomname

    for a in range(len(ann_masks)):
        ann_mask = ann_masks[a]['mask']
        object_filter = ann_mask[keypoints[:,0].T,keypoints[:,1].T]
        np_obj_pts = keypoints[np.where(object_filter==1)[0]].numpy()

        obj_id = str(ann_masks[a]['id']) 
        x = torch.mean(relative_distance[0][np.where(ann_mask==1)])
        y = torch.mean(relative_distance[1][np.where(ann_mask==1)])
        z = torch.mean(relative_distance[2][np.where(ann_mask==1)])

        image_objects['relative_distance'].append(torch.tensor([x,y,z]))
        image_objects['points'].append(keypoints[np.where(object_filter==1)[0]].float())
        image_objects['descs'].append(descriptors[np.where(object_filter==1)[0]].float())
        image_objects['ids'].append(obj_id) 
    
    return image_objects


def preprocess(configs):

    base_dir = configs["base_dir"]
    datasets = configs["datasets"]
    object_location =configs["object_location"]
    pkl_path = configs["db_path"]
    save_pkl = configs["save_pkl"]
    split = configs["split"]
    split_size = configs["split_size"]
    
    if split:
        test_data = {}
        test_data["images"] = []
        train_data = {}
        train_data["images"] = []
    else:
        data = {}
        data["images"] = []

    print("Preprocessing Data")
    for dataset in datasets:
        print("Dataset Name = ", dataset)
        dataset_path = os.path.join(base_dir, dataset)

        for scene in os.listdir(dataset_path):
            print("Scene Name = ", scene)
            scene_path = os.path.join(dataset_path, scene)

            for room_name in os.listdir(os.path.join(scene_path, "rooms")):
                print("Room Name = ", room_name)
                room_path = os.path.join(scene_path, "rooms", room_name)
                raw_data_folder = os.path.join(room_path, "raw_data/")
                points_dir = os.path.join(room_path, "points/")
                if not os.path.isdir(points_dir):
                    print("Points not available !!!")
                
                ids = ids_from_folder(raw_data_folder)
                
                if split:
                    split_value = int(len(ids)*split_size)
                    np.random.shuffle(ids)
                    train_ids, test_ids = ids[:split_value], ids[split_value:]

                    for id in train_ids:
                        room = (dataset+"_"+scene+"_"+room_name)
                        image = get_raw_data(raw_data_folder,points_dir,id,room,configs)

                        # The value of image is zero if Image is not good.
                        if image == 0 :
                            print("Image is not good : ", raw_data_folder , id)
                            remove_image(room_path,id)
                            continue

                        image_objects = get_image_objects(image)
                        train_data["images"].append(image_objects)

                    for id in test_ids:
                        room = (dataset+"_"+scene+"_"+room_name)
                        image = get_raw_data(raw_data_folder,points_dir,id,room,configs)

                        # The value of image is zero if Image is not good.
                        if image == 0 :
                            print("Image is not good : ", raw_data_folder , id)
                            remove_image(room_path,id)
                            continue

                        image_objects = get_image_objects(image)
                        test_data["images"].append(image_objects)

                else:
                    for id in ids:
                        room = (dataset+"_"+scene+"_"+room_name)
                        image = get_raw_data(raw_data_folder,points_dir,id,room,configs)

                        # The value of image is zero if Image is not good.
                        if image == 0 :
                            print("Image is not good : ", raw_data_folder , id)
                            remove_image(room_path,id)
                            continue

                        image_objects = get_image_objects(image)
                        data["images"].append(image_objects)

    if split:
        if save_pkl:
            with open(pkl_path.replace(".pkl","_train.pkl"), "wb") as out_file:
                pickle.dump(train_data, out_file)
            with open(pkl_path.replace(".pkl","_test.pkl"), "wb") as out_file:
                pickle.dump(test_data, out_file)

        
        return train_data,test_data


    else:
        if save_pkl:
            with open(os.path.join(pkl_path,str(configs["K"]) + ".pkl"), "wb") as out_file:
                pickle.dump(data, out_file)
        
        return data


def main():
    parser = argparse.ArgumentParser(description="Training AirLoc")
    parser.add_argument(
        "-c", "--config_file",
        dest = "config_file",
        type = str, 
        default = ""
    )
    args = parser.parse_args()

    config_file = args.config_file
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)

    preprocess(configs)
    
if __name__ == "__main__":
    main()