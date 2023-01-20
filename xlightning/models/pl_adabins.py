#============ DPT ================#
from xlightning.models.base import BaseModel
import torch.nn as nn
import torch


class Scale(nn.Module):
    def __init__(self,**cfg):
        super.__init__(self)
        pass

    def forward(self,x):

        return x#/1000.

import os


def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    model.load_state_dict(modified)
    return model, optimizer, epoch

import xlightning.models.adabins as ada
class AdaBins(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        _,parent_parser = AdaBins.add_base_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("adabins")   
        parser.add_argument("--n_bins", type=int, default=256)
        parser.add_argument("--norm", type=str, default="linear")
        return parent_parser

    def __init__(self,**cfg):
        BaseModel.__init__(self,**cfg)

        class args:
            n_bins = cfg['n_bins']
            min_depth = cfg['min_dist']
            max_depth= cfg['max_dist']
            norm = cfg['norm']
            
        self.model = ada.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
        
        if self.hparams['pretrained_kitti']:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_weight = torch.load('xlightning/models/depth/adabins/weights/AdaBins_kitti.pt',map_location=device)
            m = self.model
            self.model = load_checkpoint('xlightning/models/depth/adabins/weights/AdaBins_kitti.pt', m, optimizer=None)[0]

        if self.hparams['freeze_encoder']:
            for p in self.model.encoder.parameters():
                p.requires_grad=False
                    
        
    def _forward(self,x,**kwargs):
      
        bins_edge , upsampled_logits = self.model(x)

        return {'output':upsampled_logits,'others':[bins_edge]}




      
        
