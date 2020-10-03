import os
import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
#import pretrainedmodels

from ..ot_utils.rhm_map import rhm, rhm_single

import pdb


class AlignMem(nn.Module):
    def __init__(self, config):
        super(AlignMem, self).__init__()

        self.num_classes = config.numcls + 1
        self.pick_num = config.bank_pick_num
        self.otmap_thresh = config.otmap_thresh
        self.context_bank_num = 5
        self.structure_bank_num_max = config.otmap_struct_max 
        self.dim = config.bank_dim
        self.forget_para = 0.8

        self.feat_bank = torch.zeros(self.num_classes, self.dim, self.structure_bank_num_max).cuda()
        self.bank_confidence_transport = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()
        self.bank_confidence = torch.zeros(self.num_classes).cuda()
        self.context_bank = torch.zeros(self.num_classes, self.context_bank_num, self.dim).cuda()
        self.structure_bank = torch.zeros(self.num_classes, self.structure_bank_num_max, self.dim).cuda()
        self.context_bank = torch.zeros(self.num_classes, self.dim).cuda()

        self.update_feat_bank = torch.zeros(self.num_classes, self.dim, self.structure_bank_num_max).cuda()
        self.update_bank_confidence_transport = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()
        self.update_bank_confidence = torch.zeros(self.num_classes).cuda()
        self.update_bank_position = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()
        self.update_context_bank = torch.zeros(self.num_classes, self.dim).cuda()

        self.debug_img_list = ['' for x in range(200)]

        self.cos_sim = nn.CosineSimilarity(dim=2)
        self.cos_sim_1 = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.debug = False
        self.debug_save_num = 0

        self.feat_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.feat_pooling = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.feature = None
        self.gradient = None
        self.handlers = []

        self.relu = nn.ReLU() 

    def update_bank(self):
        self.feat_bank = self.update_feat_bank
        self.bank_confidence_transport = self.update_bank_confidence_transport
        self.bank_confidence = self.update_bank_confidence 
        self.context_bank = self.update_context_bank


    def operate_ot(self, candidates, candi_val, bank_feat, bank_confidence):
        re_candi = candidates.permute(0, 2, 1).cuda()
        re_bank = bank_feat.permute(0, 2, 1).cuda()

        bs, choose, dim = re_bank.size()
        re_candi = re_candi.repeat(1, choose, 1)
        re_bank = re_bank.repeat(1, 1, choose).view(bs, choose*choose, dim)

        simmap = self.cos_sim(re_candi, re_bank).view(bs, choose, choose)
        simmap = simmap.permute(0, 2, 1)
        votes = rhm(simmap, candi_val, bank_confidence.cuda()) 

        return votes


    def operate_single_ot(self, candi, candi_val, bank_feat, bank_confidence):
        re_candi = candi.permute(1, 0).cuda()
        re_bank = bank_feat.permute(1, 0).cuda()
        
        re_candi = re_candi.repeat(self.structure_bank_num_max, 1)
        re_bank = re_bank.repeat(1, self.structure_bank_num_max).view(self.structure_bank_num_max * self.structure_bank_num_max, self.dim)
        simmap = self.cos_sim_1(re_candi, re_bank).view(self.structure_bank_num_max, self.structure_bank_num_max)
        vote = rhm_single(simmap, candi_val, bank_confidence.cuda())
        return vote 


    def correct_forward(self, pick_pos, pick_val, feat, label, return_map=True):
        dim, hei, wei = feat.size()
        feat_view = feat.view(dim, -1)
        pick_feat = feat_view[:, pick_pos]

        otmap = self.operate_single_ot(pick_feat, pick_val, self.feat_bank[label], self.bank_confidence_transport[label]) 
        if return_map:
            return otmap

        aligned_feat = torch.matmul(otmap, pick_feat) 
        return [aligned_feat, self.feat_bank[label]]

        

    def error_forward(self, feat, label):
        if self.context_bank[label].sum() == 0:
            return None, None
        else:
            return feat, self.context_bank[label]


    def proc(self, scores, labels, feat, img_names=None):
        scores = self.softmax(scores)
        pred_val, pred_pos = torch.max(scores, 1) 
        pred_feat = feat# * weight[pred_pos].unsqueeze(2).unsqueeze(2)
        bs, dim, hei, wei = feat.size()
        re_feat = feat.view(bs, dim, -1)
        feat_hm = F.normalize(self.relu(feat).sum(1))
        feat_hm_view = feat_hm.view(bs, -1)
        feat_view = feat.view(bs, self.dim, -1)
        feat_norm = feat_hm.view(bs, -1).mean(1).unsqueeze(1)
        #hm_ind = torch.nonzero(feat_hm_view > feat_norm)
        
        correct_judge = (pred_pos == labels)
        error_judge = (pred_pos != labels)

        update_judge = (pred_val.cpu() - self.bank_confidence[labels].cpu()) > 0.1
        forward_judge = (self.bank_confidence[labels].cpu() - pred_val.cpu()) > 0.1
        bank_judge = (self.bank_confidence[labels].cpu() != 0).cuda()
        pred_bank_judge = (self.bank_confidence[pred_pos].cpu() != 0).cuda()
        bg_judge = (labels != self.num_classes) + (pred_pos != self.num_classes)

        update_judge = correct_judge * update_judge.cuda()
        update_judge *= bg_judge
        update_ind = torch.nonzero(update_judge).squeeze(1)

        forward_judge = correct_judge * forward_judge.cuda() # + error_judge
        forward_judge *= bg_judge 
        forward_correct_ind = torch.nonzero(forward_judge * bank_judge).squeeze(1)

        error_judge *= bank_judge
        error_judge *= pred_bank_judge
        error_judge *= bg_judge
        forward_error_ind = torch.nonzero(error_judge).squeeze(1)

        counter = 0
        otmap_gather = []
        error_context_gather = []
        bank_context_gather = []
        err_bank_context_gather = []
        for counter in range(len(pred_pos)):
            cur_hm_ind = torch.nonzero(feat_hm_view[counter] > feat_norm[counter]).squeeze(1)
            #pick_num = len(cur_hm_ind) if len(cur_hm_ind) < self.structure_bank_num_max else self.structure_bank_num_max 
            pick_num = self.structure_bank_num_max 
            pick_val, pick_pos = torch.topk(feat_hm_view[counter], pick_num) 
            if (pred_pos[counter] == labels[counter]):
                if counter in update_ind:
                    self.update_feat_bank[labels[counter].long()] = re_feat[counter][:, pick_pos].detach()
                    self.update_bank_confidence_transport[labels[counter].long()] = pick_val.detach()
                    self.update_bank_confidence[labels[counter]] = pred_val[counter]
                    self.update_bank_position[labels[counter]] = pick_pos
                    if img_names is not None:
                        self.debug_img_list[labels[counter]] = img_names[counter]
                    ct_pfeat = re_feat[counter][:, pick_pos[0]].detach()
                    ct_bfeat = self.context_bank[labels[counter].long()] 
                    ct_bfeat = self.forget_para*ct_pfeat + (1 - self.forget_para)*ct_bfeat
                    self.update_context_bank[labels[counter].long()] = ct_bfeat 
        
                elif counter in forward_correct_ind: 
                    otmap = self.correct_forward(pick_pos, pick_val, feat[counter], labels[counter])
                    otmap_gather.append(otmap)
            else:
                #ct_pfeat, ct_bfeat = self.error_forward(feat[counter], labels[counter])
                if self.context_bank[labels[counter]].sum() != 0:
                    ct_pfeat = re_feat[counter][:, pick_pos[0]]
                    error_context_gather.append(ct_pfeat)
                    bank_context_gather.append(self.context_bank[labels[counter]])
                    err_bank_context_gather.append(self.context_bank[pred_pos[counter]])
        return otmap_gather, error_context_gather, bank_context_gather, err_bank_context_gather


    def heatmap_debug_plot(self, heatmap, img_names, prefix):
        import cv2
        data_root = '../dataset/CUB_200_2011/dataset/data/'
        save_folder = './vis_tmp_save'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        counter = 0
        for sub_hm, img_name in zip(heatmap, img_names):
            np_hm = sub_hm.detach().cpu().numpy()
            np_hm -= np.min(np_hm)
            np_hm /= np.max(np_hm)  
            np_hm = np.uint8(255 * np_hm)
            np_hm = cv2.applyColorMap(np_hm, cv2.COLORMAP_JET)
            re_hm = cv2.resize(np_hm, (300, 300))
            
            raw_img = cv2.imread(os.path.join(data_root, img_name))
            re_img = cv2.resize(raw_img, (300, 300))

            canvas = np.zeros((300, 610, 3))
            canvas[:, :300, :] = re_img
            canvas[:, 310:, :] = re_hm
            save_name = img_name.split('/')[-1][:-4]
            cv2.imwrite(os.path.join(save_folder, prefix + '_' + save_name + '_' + str(counter) + '_heatmap_cmp.png'), canvas)
            counter += 1







