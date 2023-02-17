#!/usr/bin/env python3
import torch
import torch.nn as nn
import math
from model.geometry.attention import GraphAtten
from model.netvlad import NetVLAD


class EdgeDescriptor(nn.Module):
  def __init__(self, config):
    super(EdgeDescriptor, self).__init__()
    
    points_encoder_dims = config['points_encoder_dims']
    nfeat = config['descriptor_dim']
    graph_model = config['graph_model']
    nhid = config['hidden_dim']
    dropout = config['dropout']
    nheads = config['nheads']
    nout = config['nout']
    self.points_encoder = PointsEncoder(points_encoder_dims)
    self.graph_model  = graph_model

    if graph_model == "gat":
      self.gnn = GAT(nfeat, nhid, nout, dropout)
    elif graph_model == "gcn":
      self.gnn = GCN(nfeat, nhid, nout, dropout)
    elif graph_model == "netvlad":
      self.gnn = NetVLAD(8, 256)
  
  def forward(self,batch_descs):
    '''
    inputs:
    batch_points: List[Tensor], normalized points, each tensor belonging to an object
    batch_descs: List[Tensor], local feature descriptors, each tensor belonging to an object
    batch_adj: List[Tensor], adjacency matrix corresponding to the triangulation based object points graph
    return_features: bool, return node-wise graph features
    '''


    batch_features = []
    for descs in  batch_descs:
        # num_edges = descs.shape[0]
        # edges = descs.unsqueeze(0) - descs.unsqueeze(1)
        # edges = edges[torch.triu(torch.ones(num_edges,num_edges),diagonal =1)==1]
        # edges = torch.abs(edges)
        
        # print(edges)
        edges = self.points_encoder(descs)
        num_edges = edges.shape[0]
        encoded_edges = edges.unsqueeze(0) - edges.unsqueeze(1)
        encoded_edges = encoded_edges[torch.triu(torch.ones(num_edges,num_edges),diagonal =1)==1]
        
        if self.graph_model == "netvlad":
            encoded_edges = encoded_edges.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # reshape, N * C ---> 1 * C * N * 1
        output_edges= self.gnn(encoded_edges)
      
        # output_edges = nn.functional.normalize(output_edges.squeeze(), dim =-1)
        batch_features.append(output_edges.squeeze())
    
    batch_features = torch.stack(batch_features)
    batch_features = nn.functional.normalize(batch_features.squeeze(), dim =-1)
    
    return batch_features


class PointsEncoder(nn.Module):
  def __init__(self, dims):
    super(PointsEncoder, self).__init__()  
    layers = torch.nn.ModuleList([])
    for i in range(len(dims)-1):
      layers.append(nn.Linear(dims[i], dims[i+1]))
      if i != len(dims)-2:
        # layers.append(nn.BatchNorm1d((dims[i+1])))
        layers.append(nn.ReLU())

    self.layers = layers
    for i, layer in enumerate(self.layers):
      self.add_module('point_encoder{}'.format(i), layer)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = nn.functional.normalize(x, p=2, dim=-1)
    return x


# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=8):
#         '''
#         GAT: Graph Attention Network, ICLR 2018
#         https://arxiv.org/pdf/1710.10903.pdf
#         '''
#         super().__init__()
#         self.atten1 = GraphAtten(nfeat, nhid, nhid, alpha, nheads)
#         self.atten2 = GraphAtten(nhid, nhid, nclass, alpha, nheads)
#         # self.acvt = nn.Sequential(nn.ReLU(),nn.Dropout(dropout))
#         self.tran1 = nn.Linear(nclass, nclass)
#         self.relu = nn.LeakyReLU(alpha)
#         # self.linear = nn.Linear(nclass, nclass)


#     def forward(self, x):
#         x = self.atten1(x)
#         # x = self.relu(self.tran1(x))
#         x = self.atten2(x)
#         # x = self.relu(self.tran1(x))
#         x = torch.mean(x, 0)
#         x = self.tran1(x)
#         return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=8):
        '''
        GAT: Graph Attention Network, ICLR 2018
        https://arxiv.org/pdf/1710.10903.pdf
        '''
        super().__init__()
        self.attns = torch.nn.ModuleList([GraphAttn(nfeat, nhid, dropout, alpha) for _ in range(nheads)])
        for i, attention in enumerate(self.attns):
            self.add_module('attention_{}'.format(i), attention)

        self.attn = GraphAttn(nhid * nheads, nclass, dropout, alpha)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.acvt = nn.LeakyReLU()  #ELU earlier
        self.linear = nn.Linear(nclass, nclass)


    def forward(self, x):
        # x = torch.cat([attn(self.dropout1(x)) for attn in self.attns], dim=1)
        # x =  self.attn(self.dropout2(self.acvt(x)))
        
        x = torch.cat([attn(x) for attn in self.attns], dim=1)
        x =  self.attn(self.acvt(x))
        
        x = torch.mean(x, 0)
        x = self.linear(x)
        return x

class GraphAttn(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, bias=False)
        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=1), nn.Dropout(dropout))
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        h = self.tran(x)
        e = self.att1(h).unsqueeze(0) + self.att2(h).unsqueeze(1)
        e = self.leakyrelu(e.squeeze())
        return e @ h