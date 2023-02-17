#!/usr/bin/env python3

import math
import torch
import torch.nn as nn


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=8):
        '''
        GAT: Graph Attention Network, ICLR 2018
        https://arxiv.org/pdf/1710.10903.pdf
        '''
        super().__init__()
        self.attns = [GraphAttn(nfeat, nhid, dropout, alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attns):
            self.add_module('attention_{}'.format(i), attention)

        self.attn = GraphAttn(nhid * nheads, nclass, dropout, alpha)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.acvt = nn.ELU()

    def forward(self, x, adj):
        x = torch.cat([attn(self.dropout1(x), adj) for attn in self.attns], dim=1)
        return self.attn(self.dropout2(self.acvt(x)), adj)


class GraphAttn(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, bias=False)
        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=1), nn.Dropout(dropout))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        h = self.tran(x)
        e = self.att1(h).unsqueeze(0) + self.att2(h).unsqueeze(1)
        e = self.leakyrelu(e.squeeze())
        e[adj.to_dense()<=0] = -math.inf # only neighbors
        return self.norm(e) @ h
