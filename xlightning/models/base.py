from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics import  MetricCollection
from xlightning.metrics import * 
from xlightning.losses import *
#======REFERENCES
# https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen
# https://arxiv.org/pdf/2012.08270v2.pdf 
# https://www.zhuanzhi.ai/paper/a009768e44586337a4fb7356f1aacdf5
# https://arxiv.org/pdf/2201.07436v2.pdf
# https://github.com/vinvino02/GLPDepth
# https://arxiv.org/pdf/2011.14141v1.pdf


class BaseModel(LightningModule):


    def add_base_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("basemodel")
        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--loss", type=str, default="scale")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--l2_reg", type=float, default=5e-4)
        parser.add_argument("--min_dist", type=float, default=0.1)
        parser.add_argument("--max_dist", type=float, default=80)
        parser.add_argument("--scale_invariant_ratio", type=float, default=.5)
        parser.add_argument("--loss_cut", type=float, default=.8)      
        parser.add_argument("--pretrained_dataset", type=str, default='midas')


        parser.add_argument("--resize_logits", action='store_true')      
        parser.add_argument("--extend_3d", action='store_true')
        parser.add_argument("--no_disparity", action='store_true')
        parser.add_argument("--log_scale",action='store_true')        
        parser.add_argument("--no_pretrained_depth", action='store_true')    
        parser.add_argument("--freeze_encoder",action='store_true') 
        parser.add_argument("--add_dropout", action='store_true')  
        parser.add_argument("--nearest_up",action='store_true')   
        parser.add_argument("--no_pred_log",action='store_true')  
        parser.add_argument("--disp_scale", type=float, default=1.)
        parser.add_argument("--disp_shift", type=float, default=0.)      

        #parser.add_argument("--resize_logits", type=(lambda x:(x).lower()=='true'), default=True)      
        #parser.add_argument("--extend_3d", type=(lambda x:(x).lower()=='true'), default=False)
        #parser.add_argument("--disparity", type=(lambda x:(x).lower()=='false'), default=True)
        #parser.add_argument("--log_scale", type=(lambda x:(x).lower()=='true'), default=False)        
        #parser.add_argument("--pretrained_depth", type=(lambda x:(x).lower()=='true'), default=False)    
        #parser.add_argument("--freeze_encoder", type=(lambda x:(x).lower()=='true'), default=False) 
        #parser.add_argument("--add_dropout", type=(lambda x:(x).lower()=='true'), default=False)  
        #parser.add_argument("--nearest_up", type=(lambda x:(x).lower()=='true'), default=False)   
        #parser.add_argument("--pred_log", type=(lambda x:(x).lower()=='true'), default=False) 

        
        #parser.add_argument("--input_size", type=int, default=384) 
        return parser,parent_parser

    def __init__(self,**cfg):
        super().__init__()
        cfg['input_size']=cfg['SIZE']
        self.save_hyperparameters(cfg)

        train_metrics = MetricCollection({        
            'mae':MeanAbsoluteError(cfg=cfg,dist_sync_on_step=True),
            'mae_5':MeanAbsoluteError(cfg=cfg,dist_sync_on_step=True,cutoff=5.),
            'mae_10':MeanAbsoluteError(cfg=cfg,dist_sync_on_step=True,cutoff=10.),
            'mae_15':MeanAbsoluteError(cfg=cfg,dist_sync_on_step=True,cutoff=15.),
            })
        
        test_metrics = MetricCollection({        
            'mae':MeanAbsoluteError(cfg=cfg,dist_sync_on_step=True),
            'mae_5':MeanAbsoluteError(cfg=cfg,dist_sync_on_step=True,cutoff=5.),
            'mae_10':MeanAbsoluteError(cfg=cfg,dist_sync_on_step=True,cutoff=10.),
            'mae_15':MeanAbsoluteError(cfg=cfg,dist_sync_on_step=True,cutoff=15.),
            'rmse':RootMeanSquaredError(cfg=cfg,dist_sync_on_step=True),
            'rmse_5':RootMeanSquaredError(cfg=cfg,dist_sync_on_step=True,cutoff=5),
            'rmse_10':RootMeanSquaredError(cfg=cfg,dist_sync_on_step=True,cutoff=10.),
            'rmse_15':RootMeanSquaredError(cfg=cfg,dist_sync_on_step=True,cutoff=15.),
            'mse':MeanSquaredError(cfg=cfg,dist_sync_on_step=True),
            'mse_5':MeanSquaredError(cfg=cfg,dist_sync_on_step=True,cutoff=5.),
            'mse_10':MeanSquaredError(cfg=cfg,dist_sync_on_step=True,cutoff=10.),
            'mse_15':MeanSquaredError(cfg=cfg,dist_sync_on_step=True,cutoff=15.),
            'd1':d1(cfg=cfg,dist_sync_on_step=True),
            'd2':d2(cfg=cfg,dist_sync_on_step=True),
            'd3':d3(cfg=cfg,dist_sync_on_step=True),
            'd1_5':d1(cfg=cfg,dist_sync_on_step=True,cutoff=5.),
            'd2_5':d2(cfg=cfg,dist_sync_on_step=True,cutoff=5.),
            'd3_5':d3(cfg=cfg,dist_sync_on_step=True,cutoff=5.),
            'd1_10':d1(cfg=cfg,dist_sync_on_step=True,cutoff=10.),
            'd2_10':d2(cfg=cfg,dist_sync_on_step=True,cutoff=10.),
            'd3_10':d3(cfg=cfg,dist_sync_on_step=True,cutoff=10.),
            'd1_15':d1(cfg=cfg,dist_sync_on_step=True,cutoff=15.),
            'd2_15':d2(cfg=cfg,dist_sync_on_step=True,cutoff=15.),
            'd3_15':d3(cfg=cfg,dist_sync_on_step=True,cutoff=15.),
            'log_rmse':LogRootMeanSquaredError(cfg=cfg,dist_sync_on_step=True),
            'mae_rel':MeanAbsoluteErrorRel(cfg=cfg,dist_sync_on_step=True),
            'mse_rel':MeanSquaredErrorRel(cfg=cfg,dist_sync_on_step=True),
            'silog':SiLog(cfg=cfg,dist_sync_on_step=True),
            'log10':log10(cfg=cfg,dist_sync_on_step=True)})
 
        self.train_metrics = train_metrics.clone(prefix='{}/train_'.format(self.hparams['experiment_version']))
        self.valid_metrics = test_metrics.clone(prefix='{}/valid_'.format(self.hparams['experiment_version']))
        self.test_metrics = test_metrics.clone(prefix='{}/test_'.format(self.hparams['experiment_version']))

        self.lr = self.hparams['lr']

        self.loss_fn =  {'scale':ScaleInvariantLoss, 'berHu': berHu, 'l1':L1Loss}[self.hparams['loss']](**self.hparams)#

        self.scale = nn.Identity()

    def forward(self,x,**kwargs):

        output = self._forward(x,**kwargs)

        return {'upsampled_logits':output['output'],'others': output['others']}
        
    def configure_optimizers(self):
        parameters = [
                {'params': self.parameters(),'lr':self.hparams['lr']}
                
            ]
     
        #
        if self.hparams['optimizer']=='sgd':
            print('SGD OPtimizer')
            #optimizer = SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'],momentum=.9)
            optimizer = torch.optim.SGD(parameters, lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'],momentum=.9)

        elif self.hparams['optimizer']=='adam':
            print('ADAM OPtimizer')
            optimizer = torch.optim.Adam(parameters, lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])
            #optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])

        elif self.hparams['optimizer']=='adamW':
            print('ADAMW OPtimizer')
            #optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])
            optimizer = torch.optim.AdamW(parameters, lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])

        else:
            raise

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
            mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel',\
                 cooldown=0, min_lr=0, eps=1e-08, verbose=False)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": self.hparams['monitor']}

    def inv_map(self,x):
        x = 1/(x+.01) - .01
        return x

    def training_step(self, batch, batch_idx):
        sample = batch
        y_hat = self(sample['processed_image'])
      
        logits = y_hat['upsampled_logits']
        labels = sample['processed_mask']

        loss = self.loss_fn(logits, labels)
        
        if not self.hparams['no_pred_log']:
            logits = torch.e**logits

        if not self.hparams['no_disparity']:
            #logits = logits * self.hparams['disp_scale'] + self.hparams['disp_shift']
            mask = logits == 0
            logits[~mask] = 1/logits[~mask]
            #logits[mask] = self.hparams['max_dist']-1

        metrics = self.train_metrics(logits,labels)
        
        self.log('{}/train_loss'.format(self.hparams['experiment_version']),loss['total_loss'],sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)

        
        metrics_dict = {}
        for k,v in self.train_metrics.items():
            self.log('{}'.format(k),v)
            self.log('hp/{}'.format(k.split('/')[1]),v.compute(),sync_dist=True)
        
        return {'loss':loss['total_loss'],'y_hat':logits.detach()}

    def validation_step(self, batch, batch_idx):
        sample = batch
        y_hat = self(sample['processed_image'])
        
        logits = y_hat['upsampled_logits']
        labels = sample['processed_mask']

        val_loss = self.loss_fn(logits, labels)#,weight=sample['weights'])

        if not self.hparams['no_pred_log']:
            logits = torch.e**logits

        if not self.hparams['no_disparity']:
            #logits = logits * self.hparams['disp_scale'] + self.hparams['disp_shift']
            mask = logits == 0
            logits[~mask] = 1/logits[~mask]

        metrics = self.valid_metrics(logits,labels)             
        
        self.log('{}/val_loss'.format(self.hparams['experiment_version']),val_loss['total_loss'],sync_dist=True, on_step=False,on_epoch=True,prog_bar=True)

        metrics_dict = {}
        for k,v in self.valid_metrics.items():
            self.log('{}'.format(k),v)
            self.log('hp/{}'.format(k.split('/')[1]),v.compute(),sync_dist=True)


        return {'val_loss':val_loss['total_loss'],'y_hat':logits.detach()}

    def test_step(self, batch, batch_idx):
        sample = batch
        y_hat = self(sample['processed_image'])

        logits = y_hat['upsampled_logits']
        labels = sample['processed_mask']



        test_loss = self.loss_fn(logits, labels,**{'texture_logits':texture_logits,'texture_labels':texture_labels})#,weight=sample['weights'])
        
   
        with torch.no_grad():
            metrics = self.test_metrics(logits,labels)
            
        self.log('{}/test_loss'.format(self.hparams['experiment_version']),test_loss['total_loss'],on_step=True,on_epoch=True,prog_bar=True)

        metrics_dict = {}
        for k,v in self.test_metrics.items():
            self.log('hp/{}'.format(k.split('/')[1]),v.compute(),on_step=True,on_epoch=True,prog_bar=True)

        return {'test_loss':test_loss['total_loss'],'y_hat':logits.detach()}



