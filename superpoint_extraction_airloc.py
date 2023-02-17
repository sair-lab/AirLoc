import os
import argparse
import yaml
from tqdm import tqdm

import torch
import cv2
from model.build_model import build_superpoint_model
from model.inference import superpoint_inference

def find(lst, key, value):
    ind = []
    id = []
    for i, dic in enumerate(lst):
        if value in dic[key][0]:
            ind.append(i)
            id.append(lst[i]['id'])
    return ind, id

def inference(configs):
    ## data cofig
    data_config = configs['data']
    ## superpoint model config
    superpoint_model_config = configs['model']['superpoint']
    detection_threshold = superpoint_model_config['detection_threshold']
    ## others
    configs['num_gpu'] = [0]
    configs['public_model'] = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_dir = configs['img_data_path']
    datasets = configs['datasets']
    
    # model
    superpoint_model = build_superpoint_model(configs)

    for dataset in datasets:
        print("Dataset Name = ", dataset)
        dataset_path = os.path.join(base_dir, dataset)

        for scene in os.listdir(dataset_path):
            print("Scene Name = ", scene)
            scene_path = os.path.join(dataset_path, scene)
            
            for room_name in os.listdir(os.path.join(scene_path,"rooms")):
                print("Room Name = " , room_name)
                room_path = os.path.join(scene_path, "rooms", room_name)
                raw_data_folder = os.path.join(room_path,"raw_data/") 
                points_dir =  os.path.join(room_path,"points") 
                if not os.path.isdir(points_dir):
                    os.mkdir(points_dir)
                for img_name in tqdm(os.listdir(raw_data_folder)) :    
                    if img_name.endswith("rgb.png"):
                        img_path = os.path.join(raw_data_folder,img_name)
                        
                        data = {}
                        src = cv2.imread(img_path)
                        image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                        image = cv2.merge([image, image, image])
                        image = torch.from_numpy(image).type(torch.float32).to(device)
                        image = image.permute(2,0,1)
                        image /= 255
                        data['image'] = [image]
                        data['image_name'] = [str(img_name.split('_')[0])]
                        
                        with torch.no_grad():
                            result = superpoint_inference(superpoint_model, data, data_config, detection_threshold, points_dir)
                    


def main():
    parser = argparse.ArgumentParser(description="SuperPoint Feature Extraction")
    parser.add_argument(
        "-c", "--config_file",
        dest = "config_file",
        type = str, 
        default = ""
    )
    parser.add_argument(
        "-g", "--gpu",
        dest = "gpu",
        type = int,
        default = 1
    )
    args = parser.parse_args()

    config_file = args.config_file
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)
    configs['use_gpu'] = args.gpu

    inference(configs)

if __name__ == "__main__":
    main()