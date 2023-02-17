import os
import pickle
import yaml
import argparse
from datetime import datetime

import sys
sys.path.append('.')

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.spatial.distance import cdist


from model.build_model import build_netvlad, build_airloc_edge
from model.geometry.edge_loss import EdgeLoss
from datasets.mp3d_airloc.mp3d_triplet_v3_edge import mp3d
from datasets.utils.batch_collator import eval_custom_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.generate_reference import generate, points_to_obj_desc


import time
from statistics import mean
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

def preprocess(train_loader, netvlad_model, method): 
    points = []
    for step,anchor_pts in enumerate(tqdm(train_loader)):
        batch_points, query = points_to_obj_desc(anchor_pts,netvlad_model,method)
        if query == None:
            continue
        query_rooms = [obj[0]["room_image_name"][0] for obj in anchor_pts]
        
        points.append([batch_points,query_rooms])
    return points

def get_connection(ref,qry):
    con = torch.zeros(len(ref),len(qry))
    for i in range(len(ref)):
        for j in range(len(qry)):
            if ref[i]==qry[j]:
                con[i,j]=1
    con.requires_grad = True
    return con

def accuracy(ref,anc,conns):
    similarity = torch.einsum('nd,dm->nm', ref, anc.t())
    # print(similarity)
    mInds = torch.argsort(similarity,axis=0)[-1:]
    mInds = mInds.reshape(conns.shape[1])
    positive = 0
    for i in range(conns.shape[1]):
        if conns[mInds[i],i] == 1:
            positive+=1

    return positive/conns.shape[1]
    
def train(configs):
    #files config
    base_dir = configs['base_dir']
    datasets = configs['datasets']
    log_dir = configs['log_dir']
    model_save_dir = configs["airloc_save_path"]
    scenes = configs["scenes"]
    
    train_config = configs['model']['airloc']
    batch_size = train_config['train']['batch_size']
    epochs = train_config['train']['epochs']
    lr = train_config['train']['lr']
    
    configs['num_gpu'] = [0]
    configs['public_model'] = 0
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = mp3d(base_dir=base_dir,test_scenes = scenes)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=eval_custom_collate,shuffle = True)
        
    netvlad_model = build_netvlad(configs)
    netvlad_model.eval()
    
    model = build_airloc_edge(configs)
    model.train()
    
    f = open(configs['database']['config_path'], 'r', encoding='utf-8')
    ref_configs = yaml.safe_load(f.read())
    ref_configs['db_path'] = configs['database']['db_path']
    ref_configs['base_dir'] = configs['database']['db_raw_path']
    ref_configs['K'] = configs['database']['K']
    ref_configs['netvlad_model_path'] = configs['netvlad_model_path'] 
    ref_configs['method'] = method = configs['method']
    ref = generate(ref_configs)
    
    rooms = []
    ref_data = []
    ref_points = []
    for key in ref.keys():
        scene =  key.split("_")[1]
        if scene in scenes:
            rooms.append(key)
            ref_data.append(ref[key][0])
            ref_points.append(ref[key][1])
        
    edge_loss = EdgeLoss(train_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'_airobj'))
    
    logdir = writer.file_writer.get_logdir()
    save_dir = os.path.join(logdir, 'saved_model')
    os.makedirs(save_dir, exist_ok=True)
    
    points = preprocess(train_loader, netvlad_model, configs['method'])
    
    train_accuracy = []
    train_loss = []
    sum_iter = 0
    prev_acc = 0
    
    for epoch in tqdm(range(epochs)):
        for batch_points,query_rooms in tqdm(points):
                  
            anc = model(batch_points)
            ref = model(ref_points)

            connections = get_connection(rooms, query_rooms)
            loss = edge_loss(ref, anc, connections)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()   
            optimizer.step()
            
            acc = accuracy(ref, anc, connections)
            train_accuracy.append(acc)
             
            sum_iter+=1
            writer.add_scalar('Train/Loss', loss, sum_iter)
            writer.add_scalar('Train/Accuracy', acc, sum_iter)
        
        print("Train_accuracy : ", mean(train_accuracy) )
        print("Train_loss : ", mean(train_loss) )
        
        torch.save(model.state_dict(), model_save_dir)

def main():
    parser = argparse.ArgumentParser(description="Training AirLoc")
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

    train(configs)
    
if __name__ == "__main__":
    main()