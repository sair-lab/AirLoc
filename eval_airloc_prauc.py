import pickle
import yaml
import argparse
from scipy.spatial.distance import cdist
import os

import sys
sys.path.append('.')

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from model.build_model import build_netvlad, build_airloc_edge, build_gcn
from datasets.mp3d_airloc.mp3d_triplet_v3_edge import mp3d
from datasets.utils.batch_collator import eval_custom_collate
from statistics import mean
from utils.generate_reference import generate,points_to_obj_desc

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

def get_connection(ref,qry):
    con = torch.zeros(len(ref),len(qry))
    for i in range(len(ref)):
        for j in range(len(qry)):
            if ref[i]==qry[j]:
                con[i,j]=1
    con.requires_grad = True
    return con

def get_pr_curve_area(pr_curve):
  '''
  pr_curve: [[p0, r0], [p1, r1]... [pn, rn]], thr: small->big, precision: small->big, recall: big->small
  '''
  area = 0.0
  for i in range(1, len(pr_curve)):
    p0, r0 = pr_curve[i-1]
    p1, r1 = pr_curve[i]

    area = area + (r0 - r1) * (p1 + p0) / 2

  return area


def pickle_write(path, dump_file):
    with open(path, 'wb') as fp:
        pickle.dump(dump_file, fp)

  
def eval(configs):
    base_dir = configs['base_dir']
    batch_size = configs['batch_size']
    scenes = configs["scenes"]
    ap_thres = configs['ap_thres']
    m = configs["ap_wght"]
    method = configs['method']

    test_dataset = mp3d(base_dir=base_dir,test_scenes = scenes)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=eval_custom_collate,shuffle = True)

    if method in ["airloc","airloc_without_edge",'netvlad_mean','netvlad']:
        model = build_netvlad(configs)
        model.eval() 

    if method in ["airloc"]:
        edge_model = build_airloc_edge(configs)
        edge_model.eval()
    
    if method in ["gcn","gcn_mean"]:
        model = build_gcn(configs)
        model.eval() 

    
    f = open(configs['database']['config_path'], 'r', encoding='utf-8')
    ref_configs = yaml.safe_load(f.read())
    ref_configs['db_path'] = configs['database']['db_path']
    ref_configs['base_dir'] = configs['database']['db_raw_path']
    ref_configs['K'] = configs['database']['K']
    ref_configs['netvlad_model_path'] = configs['netvlad_model_path'] 
    ref_configs['graph_model_path'] = configs['graph_model_path']
    ref_configs['method'] = method
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
    
    with torch.no_grad():
        test_accuracy = []

        conn = []
        sim = []
        for step, anchor_pts in enumerate(tqdm(test_loader)):
            anchor_pt, query = points_to_obj_desc(anchor_pts,model,method)
            if query == None:
                continue
            
            if method in ["airloc","airloc_without_edge",'gcn']:
                room_sim = []
                for i in range(len(ref_data)):
                    ob_sim = []
                    for j in range(len(query)):
                        prod = (ref_data[i] @ query[j].T.to(ref_data[i].device))
                        ref = torch.norm(ref_data[i], dim = 1).to(prod.device)        # Normalising
                        qry = torch.norm(query[j], dim = 1).to(prod.device)  
                        mat = ((prod/qry).T/ref).T
                        prod = torch.max(mat, dim = 0).values
                        ob_sim.append(torch.sum(prod))
                    room_sim.append(torch.stack(ob_sim))
                    
                room_sim = torch.stack(room_sim)
                room_sim = torch.nn.functional.normalize(room_sim,dim = 0)

            elif method in ["netvlad_mean",'gcn_mean','netvlad']:
                room_sim = 1-torch.cdist(torch.stack(ref_data),torch.stack(query).to(torch.stack(ref_data)),p = 2)

            # print(torch.nn.functional.normalize(room_sim,dim = 0))
            # print(room_sim)
            if method == "airloc":
                mValues, indices = torch.sort(room_sim,axis=0)

                for i, obj in enumerate(anchor_pts):
                    query_room = obj[0]["room_image_name"][0]
            
                    if (mValues[-1,i]-mValues[-2,i])> ap_thres : 
                        continue
                    else:
                        anc_e = edge_model(anchor_pt)
                        ref_e = edge_model(ref_points)
                        prod = (ref_e@ anc_e.T)
                        room_sim_ = m*torch.nn.functional.normalize(room_sim,dim = 0)+(1-m)*torch.nn.functional.normalize(prod,dim = 0).to(room_sim.device)
                        room_sim[:,i] = room_sim_[:,i]
                        
            query_rooms = [obj[0]["room_image_name"][0] for obj in anchor_pts]
            connections = get_connection(rooms, query_rooms)

            sim.append(room_sim)
            conn.append(connections)

    thrs = [float(i)/200 for i in range(201)]
    pr_curve = []   

    print("Evaluating for different thresholds")
    for thr in tqdm(thrs):
        pr_numbers = []

        for s , c in zip(sim, conn):

            match_matrix = (s.cpu() > thr).float() 
            tp = torch.sum(match_matrix * c.cpu())
            match_num = torch.sum(match_matrix).item()
            gt_num = torch.sum(c).item()

            pr_number = [tp, match_num, gt_num]
            pr_numbers.append(pr_number)
            
        pr_numbers = torch.tensor(pr_numbers)
        pr_numbers = torch.sum(pr_numbers, 0)

        TP, MatchNum, GTNum = pr_numbers.cpu().numpy().tolist()

        precision = TP / MatchNum if MatchNum > 0 else 1
        recall = TP / GTNum if GTNum > 0 else 1
        pr_curve.append([precision, recall])          
            
    area = get_pr_curve_area(pr_curve)
    print('PR-AUC(%): {:.2f}'.format(area*100))

    results = {}
    results['pr_curve'] = pr_curve
    results['area'] = area

    os.makedirs(configs['save_dir'], exist_ok=True)
    pickle_write(os.path.join(configs['save_dir'], configs['method']+'_pr_curve.pkl'), results)

def main():
    parser = argparse.ArgumentParser(description="Evaluating")
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
    
    eval(configs)
    
if __name__ == "__main__":
    main()
