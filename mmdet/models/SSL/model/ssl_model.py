import os
import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F

import pdb

class SSLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
         
        self.dim = config.bank_dim
        self.num_classes = config.numcls + 1

        self.mapping_module = nn.Sequential(
                                       nn.Conv2d(2048, 1024, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(1024, self.dim, 3, stride=1, padding=1),
                                       nn.ReLU(),)
        self.mapping_cls = nn.Sequential(
                                     nn.Conv2d(self.dim, self.dim, 3, stride=2), 
                                     nn.ReLU(),
                                     nn.Conv2d(self.dim, self.num_classes, 3, stride=2),
                                     nn.ReLU(),) 


    def forward(self, feat):
        map_feat = self.mapping_module(feat)
        map_cls = self.mapping_cls(map_feat)
        map_cls = map_cls.view(map_cls.size(0), self.num_classes, -1)
        map_cls, _ = map_cls.max(2)
        return map_feat, map_cls


