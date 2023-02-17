#!/usr/bin/env python3
import torch
import torch.nn as nn


class Descriminator(nn.Module):

    def __init__(self,nfeat):
        super().__init__()
        self.linear = nn.Linear(nfeat, nfeat)
        self.linear.weight.data = torch.zeros(nfeat,nfeat)
        
    def forward(self, ref_data, query):
        
        matches = []
        for j in range(len(query)):
            prod = (ref_data @ query[j].T)
            ref = torch.norm(ref_data, dim = 1)
            qry = torch.norm(query[j], dim = 1)
            mat = ((prod/qry).T/ref).T
            mat = torch.max(mat, dim = 0).values

            y = self.linear(query[j])
            b = torch.mean(y,0)
            c = y @ b
            c = c/c.sum()
            b = c*mat

            matches.append(b) 
        return matches
