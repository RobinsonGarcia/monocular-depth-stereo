#============ DPT ================#
from xlightning.models.depth.base import BaseDepthModel
from xlightning.losses import berHu, ScaleInvariantLoss ,L1Loss, ScaleInvariantLossGradient
from xlightning.models.depth.dpt.models import DPTSegmentationModel, DPTDepthModel
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.optim import Adam,AdamW,SGD
class Scale(nn.Module):
    def __init__(self,**cfg):
        super.__init__(self)
        pass

    def forward(self,x):

        return x#/1000.

import os
import urllib.request
def download_from_url(url,saveto):
    urllib.request.urlretrieve(url, saveto)
    pass


class DPTmodel2(BaseDepthModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--loss", type=str, default="scale")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--l2_reg", type=float, default=0)
        parser.add_argument("--dpt_model", type=str, default="dpt_hybrid_scratch")
        parser.add_argument("--min_dist", type=float, default=0.01)
        parser.add_argument("--max_dist", type=float, default=60)
        parser.add_argument("--scale_invariant_ratio", type=float, default=.5)
        parser.add_argument("--disparity", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--log_scale", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--loss_cut", type=float, default=.8)      
        parser.add_argument("--resize_logits", type=(lambda x:(x).lower()=='false'), default=True)      
        parser.add_argument("--add_texture_head", type=(lambda x:(x).lower()=='true'), default=False)   
        parser.add_argument("--extend_3d", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--features", type=int, default=256)
        parser.add_argument("--freeze_encoder", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--pretrained_kitti", type=(lambda x:(x).lower()=='true'), default=False)    
        parser.add_argument("--add_dropout", type=(lambda x:(x).lower()=='true'), default=False)    
        return parent_parser

    def __init__(self,**cfg):
        BaseDepthModel.__init__(self,**cfg)

        #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.dpt = torch.hub.load("intel-isl/MiDaS", model_type,skip_validation=True)

        self.lr = self.hparams['lr']

        self.scale = nn.Identity()

        self.loss_fn =  {'scale':ScaleInvariantLoss, 'berHu': berHu, 'l1':L1Loss,'scale_gradient':ScaleInvariantLossGradient}[self.hparams['loss']](**self.hparams)#
    

    def forward(self,x):
      
        upsampled_logits = self.dpt(x).unsqueeze(1)

        upsampled_logits = self.scale(upsampled_logits)/1000
  
        if (self.hparams['log_scale'])|('scale' in self.hparams['loss']):
            return {'upsampled_logits':upsampled_logits,'clf_logits': None}
        else:    
            upsampled_logits = torch.nn.functional.relu(upsampled_logits) + .001

        return {'upsampled_logits':upsampled_logits,'clf_logits': None}

    def configure_optimizers(self):
        parameters = [
                #{'params': self.dpt.pretrained.parameters(),'lr':self.hparams['lr']*.01},
                {'params': self.parameters(), 'lr': self.hparams['lr']},
                
            ]
     
        #
        if self.hparams['optimizer']=='sgd':
            print('SGD OPtimizer')
            #optimizer = SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'],momentum=.9)
            optimizer = SGD(parameters, lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'],momentum=.9)

        elif self.hparams['optimizer']=='adam':
            print('ADAM OPtimizer')
            optimizer = Adam(parameters, lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])
            #optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])

        elif self.hparams['optimizer']=='adamW':
            print('ADAMW OPtimizer')
            #optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])
            optimizer = AdamW(parameters, lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])

        else:
            raise

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
            mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel',\
                 cooldown=0, min_lr=0, eps=1e-08, verbose=False)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": self.hparams['monitor']}

    
        
