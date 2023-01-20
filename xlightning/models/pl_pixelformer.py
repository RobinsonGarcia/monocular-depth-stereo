#============ DPT ================#
from xlightning.models.base import BaseModel
from xlightning.models.pixelformer.networks.PixelFormer import PixelFormer
import torch.nn as nn
import torch
from torch.optim import Adam,AdamW,SGD





class PIXelFormer(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        _,parent_parser = PIXelFormer.add_base_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("pixelformer")  
        parser.add_argument("--version", type=str, default='large07')
        return parent_parser

    def __init__(self,**cfg):
        BaseModel.__init__(self,**cfg)


        class args:
            version='large07'
            pretrained= '/petrobr/algo360/current/MultiGPU-lightning/xlightning/models/depth/pixelformer/weights/swin_large_patch4_window7_224_22k.pth'if self.hparams['pretrained_kitti'] else None

        self.model = PixelFormer(version=args.version, 
                                    inv_depth=False, 
                                    min_depth=cfg['min_dist'],
                                    max_depth=cfg['max_dist'], 
                                    pretrained=args.pretrained)


    def _forward(self,x):
      
      return {'output':self.model(x),'others':[]}
 