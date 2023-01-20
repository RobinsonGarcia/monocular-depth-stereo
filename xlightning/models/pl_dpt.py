#============ DPT ================#
from xlightning.models.base import BaseModel

from xlightning.models.dpt.models import DPTSegmentationModel, DPTDepthModel

import torch.nn as nn
import torch


import os
import urllib.request
def download_from_url(url,saveto):
    urllib.request.urlretrieve(url, saveto)
    pass

download_weights = {
        "midas_v21": "xlightning/dpt/weights/midas_v21-f6b98070.pt",
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt",
    }

class DPTmodel(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        _,parent_parser = DPTmodel.add_base_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("dpt")
        parser.add_argument("--features", type=int, default=256)
        parser.add_argument("--disp_scale", type=float, default=1.)
        parser.add_argument("--disp_shift", type=float, default=0.)
        
        return parent_parser
    def __init__(self,**cfg):
        BaseModel.__init__(self,**cfg)
        if self.hparams['pretrained_depth']:
            self.hparams['model'] = '_'.join([self.hparams['model'],self.hparams['pretrained_dataset']])


        default_models = {
        "midas_v21": "/nethome/algo360/mestrado/monocular-depth-estimation/xlightning/models/dpt/weights/midas_v21-f6b98070.pt",
        "dpt_large": "/nethome/algo360/mestrado/monocular-depth-estimation/xlightning/models/dpt/weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "/nethome/algo360/mestrado/monocular-depth-estimation/xlightning/models/dpt/weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "/nethome/algo360/mestrado/monocular-depth-estimation/xlightning/models/dpt/weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "/nethome/algo360/mestrado/monocular-depth-estimation/xlightning/models/dpt/weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }
        try:
            if not os.path.isfile(default_models[self.hparams['model']]):
                download_from_url(url=download_weights[self.hparams['model']],\
                            saveto=default_models[self.hparams['model']])
        except Exception as e:
            print(e)
    
        if self.hparams['model'] == 'dpt_hybrid_nyu':
            print('DPT hybrid - nyu')
            self.model = DPTDepthModel(
                path=default_models[self.hparams['model']],
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
                **self.hparams
            )

        elif self.hparams['model'] == "dpt_hybrid_kitti":
            print('DPT hybrid - kitti')


            self.model = DPTDepthModel(
                path=default_models[self.hparams['model']],
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=False,
                enable_attention_hooks=False,
                **self.hparams
        )

        
        elif self.hparams['model'] == "dpt_large":  # DPT-Large
            print('DPT large - midas')
            self.model= DPTDepthModel(
                path=default_models[self.hparams['model']],#if self.hparams['pretrained_depth'] else None,
                backbone="vitl16_384",
                non_negative=False,
                enable_attention_hooks=False,
                **self.hparams
            )


        elif self.hparams['model'] == "dpt_hybrid":
            print('DPT hybrid - midas')
            
            self.model = DPTDepthModel(
                        path=default_models[self.hparams['model']],# if self.hparams['pretrained_depth'] else None,
                        backbone="vitb_rn50_384",
                        non_negative=False,
                        invert=False,
                        enable_attention_hooks=False,
                        **self.hparams
                    )
        else:
            raise



    def _forward(self,x,inv_map=False):


        return {'output':self.model(x).unsqueeze(1),'others':[]}

         
        
