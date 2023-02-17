import os
import pickle
import yaml
import argparse

import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
import shutil
from scipy.stats import moment

import sys
sys.path.append('.')

from datasets.preprocess_reloc_database import preprocess
from model.build_model import build_gcn, build_netvlad, build_seqnet, build_airobj ,build_airloc,build_airloc_v3


def points_to_obj_desc(batch_objects,model,method):
    batch_decs = []
    batch_points = []

    if method == "netvlad":
        for seq_objects in batch_objects:
            seq_descs = []
            for image_objects in seq_objects:
                seq_descs.append(torch.cat(image_objects['descs']))
            image_desc = model(seq_descs)
            image_desc = torch.mean(image_desc,0)
            batch_decs.append(image_desc.squeeze())

        return [None], batch_decs

    for seq_objects in batch_objects:
        object_dict = {}
        seq_descs = []
        seq_points = []
        for image_objects in seq_objects:
            
            #Takes only those objects whone no. points are more then rejection ap_thres
            iter_loc = []
            iter_points = []
            iter_desc = []
            iter_ids = []
            for i,object_points in enumerate(image_objects['descs']):
                if object_points.shape[0]>3:
                    iter_desc.append(object_points)
                    iter_points.append(image_objects['points'][i])

                    mean = torch.mean(image_objects['points'][i],0)
                    std = torch.std(image_objects['points'][i], 0)
                    u, s, v = torch.svd(image_objects['points'][i])
                    m1 = torch.tensor(moment(image_objects['points'][i], moment=1))
                    m2 = torch.tensor(moment(image_objects['points'][i], moment=2))
                    m3 = torch.tensor(moment(image_objects['points'][i], moment=3))
                    
                    iter_loc.append(torch.cat((mean,std,m1,m2,m3,s),0))
                    iter_ids.append(image_objects['ids'][i])
                    

            if len(iter_desc) == 0:
                return None,None
            
            if method in ["airloc",'airloc_without_edge','netvlad_mean']:
                object_desc = model(iter_desc)
            elif method in ["gcn",'gcn_mean']:
                object_desc = model(iter_points,iter_desc)
            
            for i,id in enumerate(iter_ids):
                if id not in object_dict.keys():
                    object_dict[id] = {}
                    object_dict[id]["descs"] = object_desc[i].unsqueeze(0)
                    object_dict[id]["points"] = iter_loc[i].unsqueeze(0)
                else:
                    object_dict[id]["descs"] = torch.cat((object_dict[id]["descs"],object_desc[i].unsqueeze(0)),0)
                    object_dict[id]["points"] = torch.cat((object_dict[id]["points"],iter_loc[i].unsqueeze(0)),0)

        for key in object_dict.keys():
            seq_descs.append(torch.mean(object_dict[key]["descs"],0).squeeze())
            seq_points.append(torch.mean(object_dict[key]["points"],0).squeeze())

        if len(seq_descs) <= 4:
            return None,None
        if method in ["airloc",'airloc_without_edge','gcn']:
            batch_decs.append(torch.stack(seq_descs))
        elif method in  ["netvlad_mean",'gcn_mean']:
            batch_decs.append(torch.mean(torch.stack(seq_descs),0))
        batch_points.append(torch.stack(seq_points))
    
    return batch_points, batch_decs

def generate_ref_filesystem(configs):
    base_dir = configs["base_dir"]
    datasets = configs["datasets"]
    # ids = configs["ids"]
    K = configs["K"]

    for dataset in datasets:
        print("Dataset Name = ", dataset)
        dataset_path = os.path.join(base_dir, dataset)
        ref_dataset_path = os.path.join(base_dir,dataset+"_ref")
        if os.path.isdir(ref_dataset_path):
            shutil.rmtree(ref_dataset_path)    #Remove if already exists
        os.makedirs(ref_dataset_path,exist_ok=True)

        for scene in os.listdir(dataset_path):
            print("Scene Name = ", scene)
            scene_path = os.path.join(dataset_path, scene)
            ref_scene_path = os.path.join(ref_dataset_path,scene)
            os.makedirs(ref_scene_path,exist_ok=True)
            ref_rooms_path = os.path.join(ref_scene_path,"rooms")
            os.makedirs(ref_rooms_path,exist_ok=True)

            for room_name in os.listdir(os.path.join(scene_path, "rooms")):
                print("Room Name = ", room_name)
                room_path = os.path.join(scene_path, "rooms", room_name)
                ref_room_path = os.path.join(ref_scene_path, "rooms", room_name)
                os.makedirs(ref_room_path,exist_ok=True)
                raw_data_folder = os.path.join(room_path, "raw_data/")
                ref_raw_data_folder = os.path.join(ref_room_path, "raw_data/")
                os.makedirs(ref_raw_data_folder,exist_ok=True)
                points_dir = os.path.join(room_path, "points/")
                ref_points_dir = os.path.join(ref_room_path, "points/")
                os.makedirs(ref_points_dir,exist_ok=True)
            
                # for id in range(K)*100: # When there is no different database (Database images are selected from main dataset itself)
                for id in range(K):
                        rgb_data = os.path.join(raw_data_folder,(str(id)+"_rgb.png"))
                        seg_data = os.path.join(raw_data_folder,(str(id)+"_instance-seg.png"))
                        depth_data = os.path.join(raw_data_folder,(str(id)+"_depth.png"))
                        points_data = os.path.join(points_dir,(str(id)+".pkl"))

                        ref_rgb_data = os.path.join(ref_raw_data_folder,(str(id)+"_rgb.png"))
                        ref_seg_data = os.path.join(ref_raw_data_folder,(str(id)+"_instance-seg.png"))
                        ref_depth_data = os.path.join(ref_raw_data_folder,(str(id)+"_depth.png"))
                        ref_points_data = os.path.join(ref_points_dir,(str(id)+".pkl"))

                        shutil.copyfile(rgb_data, ref_rgb_data)
                        shutil.copyfile(seg_data, ref_seg_data)
                        shutil.copyfile(depth_data, ref_depth_data)
                        shutil.copyfile(points_data, ref_points_data)


def generate(configs):
    print("Generating Reference Data")
    configs['num_gpu'] = [0]
    configs['public_model'] = 0
    K = configs["K"]

    if configs['method'] in ["airloc","airloc_without_edge",'netvlad_mean','netvlad']:
        model = build_netvlad(configs)
        model.eval() 
    
    if configs['method'] in ["gcn","gcn_mean"]:
        model = build_gcn(configs)
        model.eval() 


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if os.path.isfile(os.path.join(configs["db_path"] ,str(K) + ".pkl")):
        with open(os.path.join(configs["db_path"] ,str(K) + ".pkl"),'rb') as f:
            images = pickle.load(f)
    else:
        generate_ref_filesystem(configs)
        
        datasets = configs["datasets"]
        configs["datasets"] = [dataset+"_ref" for dataset in datasets]
        images = preprocess(configs)
    sorted_images = {}
    for image in images["images"]:
        if image["room_image_name"][0] not in sorted_images.keys():
            sorted_images[image["room_image_name"][0]] = []
        sorted_images[image["room_image_name"][0]].append(image)

    room_descriptors = {}
    for key in sorted_images.keys():
        pts, objs = points_to_obj_desc([sorted_images[key]],model,configs['method'] )

        key = key.replace("_ref","")
        room_descriptors[key] = [objs[0],pts[0]]
    
    # pkl_path = configs["ref_pkl_path"]
    # with open(pkl_path, "wb") as out_file:
    #     pickle.dump(room_descriptors, out_file)

    return room_descriptors


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

    generate(configs)
    
if __name__ == "__main__":
    main()
    