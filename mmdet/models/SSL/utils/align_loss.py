import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from ..utils.entropy_2d import Entropy 

import pdb

def align_loss(otmap_gather_list, 
               pred_gather_list, 
               ctbank_gather_list, 
               err_ctbank_gather_list, 
               use_structure,
               use_context,
               structure_max ):
    get_entropy = Entropy()
    get_sim_loss = nn.CosineEmbeddingLoss()
    get_smooth_l1_loss = nn.SmoothL1Loss()

    entropy_val = torch.tensor(0.0)
    map_loss = torch.tensor(0.0)
    otmap_len = len(otmap_gather_list)
    if otmap_len > 0 and use_structure:
        otmap_gather_stack = torch.stack(otmap_gather_list)
        otmap_best_label = [torch.eye(structure_max) for x in range(otmap_len)]
        otmap_best_label = torch.stack(otmap_best_label).cuda()
        otmap_best_label = Variable(otmap_best_label)

        entropy_val = get_entropy(otmap_gather_stack)
        map_loss = get_smooth_l1_loss(otmap_gather_stack, otmap_best_label) 
        #map_loss = get_ce_loss(otmap_gather_stack.view(
            
    ct_cor_loss = torch.tensor(0.0)
    ct_err_loss = torch.tensor(0.0)
    if len(pred_gather_list) > 0 and use_context:
        pred_gather_stack = torch.stack(pred_gather_list)
        ctbank_gather_stack = torch.stack(ctbank_gather_list)
        err_ctbank_gather_stack = torch.stack(err_ctbank_gather_list)

        ct_cor_loss = get_sim_loss(pred_gather_stack, ctbank_gather_stack, torch.ones(len(pred_gather_list), 1).cuda()) 
        ct_err_loss = get_sim_loss(pred_gather_stack, err_ctbank_gather_stack, torch.zeros(len(pred_gather_list), 1).cuda()) 

    return entropy_val.cuda(), map_loss.cuda(), ct_cor_loss.cuda(), ct_err_loss.cuda()


