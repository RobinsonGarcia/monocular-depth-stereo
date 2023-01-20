#============ DPT ================#
from xlightning.models.base import BaseModel
import torch.nn as nn
import torch

from xlightning.models.networks.NewCRFDepth import NewCRFDepth


def load_newcrf(model,dataset):
    print('loading pre-trained weights')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_weight = torch.load('xlightning/models/networks/weights/{}.ckpt'.format('model_kittieigen' if dataset=='kitti' else 'model_nyu'),map_location=device)
    m = torch.nn.DataParallel(model)
    m.load_state_dict(model_weight['model'])
    return m.module


class NewCRF(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        _,parent_parser = NewCRF.add_base_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("bts")  
        parser.add_argument("--encoder", type=str, default='large07')
        parser.add_argument("--bts_size", type=int, default=512)
        parser.add_argument("--focal", type=float, default=1.)
        parser.add_argument("--model_name", type=str, default='bts_eigen_v2_pytorch_densenet161')
        return parent_parser

    def __init__(self,**cfg):
        BaseModel.__init__(self,**cfg)

        class args:
            encoder = self.hparams['encoder']
            inv_depth=False
            max_depth=self.hparams['max_dist']
            pretrain=None if self.hparams['pretrained_depth'] else '/nethome/algo360/mestrado/monocular-depth-estimation/xlightning/models/networks/swin_large_patch4_window7_224_22k.pth'

        self.model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=args.pretrain,**self.hparams)
        
        if self.hparams['pretrained_depth']:
            self.model = load_newcrf(self.model,self.hparams['pretrained_dataset'])
            

        if self.hparams['freeze_encoder']:
            for p in self.model.backbone.parameters():
                p.requires_grad=False



    def _forward(self,x):
      
      return {'output':self.model(x),'others':[]}
 