from xlightning.models.base import BaseModel
import torch.nn as nn
import torch
from xlightning.models.BTS.bts import BtsModel




class BTS(BaseModel):

    @staticmethod
    def add_model_specific_args(parent_parser):
        _,parent_parser = BTS.add_base_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("bts")  
        parser.add_argument("--encoder", type=str, default='densenet161_bts')
        parser.add_argument("--bts_size", type=int, default=512)
        parser.add_argument("--focal", type=float, default=1.)
        parser.add_argument("--model_name", type=str, default='bts_eigen_v2_pytorch_densenet161')
        return parent_parser
    def __init__(self,**cfg):
        BaseModel.__init__(self,**cfg)

        class params:
            encoder = self.hparams['encoder']
            max_depth = self.hparams['max_dist']
            dataset = 'none'
            bts_size = self.hparams['bts_size']
            focal = self.hparams['focal']
            model_name = self.hparams['model_name']

        self.model = BtsModel(params,device=self.device)

        if self.hparams['pretrained_kitti']:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_weight = torch.load('xlightning/models/depth/BTS/weights/bts_eigen_v2_pytorch_densenet161/model',map_location=device)
            m = torch.nn.DataParallel(self.model)
            m.load_state_dict(state_dict=model_weight['model'])
            self.model = m.module

        self.focal = torch.tensor(self.hparams['focal'])


    def _forward(self,x):

        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est =self.model(x,self.focal)
      
        return {'output':depth_est,'others':[lpg8x8, lpg4x4, lpg2x2, reduc1x1]}
 
        
