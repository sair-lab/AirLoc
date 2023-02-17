#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
from tkinter import N
sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
  '''
  loss for object descriptor
  '''
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.p_margin = self.config['train']['positive_margin']
    self.n_margin = self.config['train']['negative_margin']
    

  def forward(self, ref,anc,conns):
    similarity = torch.einsum('nd,dm->nm', ref, anc.t())
    pos_idx = conns.to(ref.device)


    neg_idx1 = torch.ones_like(conns).to(similarity.device) - conns.to(similarity.device)
    
    # neg_similarity0 = similarity * neg_idx0.to(similarity.device)
    # value, index = neg_similarity0.topk(1, largest=True)
    # value = value.repeat(1, similarity.shape[1])
    # neg_idx1 = (neg_similarity0 == value).float()

    zero = torch.tensor(0.0, dtype=similarity.dtype, device=similarity.device)
    positive_dist = torch.max(zero, self.config['train']['positive_margin'] - similarity)
    negative_dist = torch.max(zero, similarity - self.config['train']['negative_margin'])
    
    # positive_dist = self.config['train']['positive_margin'] - similarity
    # negative_dist = similarity - self.config['train']['negative_margin']
  

    if torch.sum(pos_idx) != 0:
      ploss = torch.sum(pos_idx * positive_dist) / torch.sum(pos_idx)
    else:
      ploss = torch.tensor(0.0, dtype=similarity.dtype, device=similarity.device)
    if torch.sum(neg_idx1) != 0:
      nloss = torch.sum(neg_idx1 * negative_dist) / torch.sum(neg_idx1)
    else:
      nloss = torch.tensor(0.0, dtype=similarity.dtype, device=similarity.device)

    return ploss+ nloss
        
        
        