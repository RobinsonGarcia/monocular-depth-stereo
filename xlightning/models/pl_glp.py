#============ DPT ================#
from xlightning.models.depth.base import BaseDepthModel
from xlightning.losses import berHu, ScaleInvariantLoss ,L1Loss, ScaleInvariantLossGradient
from xlightning.models.depth.dpt.models import DPTSegmentationModel, DPTDepthModel
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.optim import Adam,AdamW,SGD

from xlightning.models.depth.glp.models.model import GLPDepth
from xlightning.models.depth.base import BaseDepthModel

from collections import OrderedDict

class GLPmodel(BaseDepthModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        
        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--loss", type=str, default="l1")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--l2_reg", type=float, default=5e-4)
        parser.add_argument("--min_dist", type=float, default=0.1)
        parser.add_argument("--max_dist", type=float, default=10)
        parser.add_argument("--scale_invariant_ratio", type=float, default=.5)
        parser.add_argument("--disparity", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--log_scale", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--loss_cut", type=float, default=.8)      
        parser.add_argument("--resize_logits", type=(lambda x:(x).lower()=='true'), default=True)    
        parser.add_argument("--add_texture_head", type=(lambda x:(x).lower()=='true'), default=False)    
        parser.add_argument("--extend_3d", type=(lambda x:(x).lower()=='true'), default=False)

        parser.add_argument("--pretrained_kitti", type=(lambda x:(x).lower()=='true'), default=False)    
        parser.add_argument("--patch_size", type=int, default=4)
        parser.add_argument("--freeze_encoder", type=(lambda x:(x).lower()=='true'), default=False)    

        
        parser.add_argument("--add_dropout", type=(lambda x:(x).lower()=='true'), default=False)  #==> NOT IMPLEMENTED ON THIS MODEL


        return parent_parser
    
    def __init__(self,**cfg):
        BaseDepthModel.__init__(self,**cfg)
        kwargs = cfg
        self.glp = GLPDepth(max_depth=cfg['max_dist'], is_train=True,**kwargs)
        
        self.lr = self.hparams['lr']

        if self.hparams['freeze_encoder']:
            for params in self.glp.encoder.parameters():
                params.requires_grad=False
            self.glp.encoder.eval()

        if self.hparams['pretrained_kitti']:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_weight = torch.load('xlightning/models/depth/glp/weights/best_model_kitti.ckpt',map_location=device)
            if 'module' in next(iter(model_weight.items()))[0]:
                    model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())

            self.glp.load_state_dict(model_weight)

        self.loss_fn =  {'scale':ScaleInvariantLoss, 'berHu': berHu, 'l1':L1Loss,'scale_gradient':ScaleInvariantLossGradient}[self.hparams['loss']](**self.hparams)#

    def forward(self,x):

        upsampled_logits = self.glp(x)
        
        #upsampled_logits = torch.nn.functional.relu(upsampled_logits) 

        return {'upsampled_logits':upsampled_logits,'clf_logits': None}


    def configure_optimizers(self):
        parameters = [
                {'params': self.parameters(),'lr':self.hparams['lr']}]

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

        #return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": self.hparams['monitor']}
        #return {"optimizer": optimizer, "monitor": self.hparams['monitor']}


        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": self.hparams['monitor']}


    def __optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # skip the first 500 steps
        if self.trainer.global_step < 100:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 100.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)       
