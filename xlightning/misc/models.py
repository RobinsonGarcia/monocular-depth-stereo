from functools import lru_cache
import pytorch_lightning as pl
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,SGD, AdamW
import torch
import torchmetrics
from torchmetrics import Accuracy, Precision , Recall, MetricCollection, F1
from xlightning.losses import FocalLossFromLogits, ScaleInvariantLoss, berHu
from xlightning.metrics import MyIoU, MeanAbsoluteError, MeanSquaredError


#======REFERENCES
# https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen
# https://arxiv.org/pdf/2012.08270v2.pdf 
# https://www.zhuanzhi.ai/paper/a009768e44586337a4fb7356f1aacdf5
# https://arxiv.org/pdf/2201.07436v2.pdf
# https://github.com/vinvino02/GLPDepth
# https://arxiv.org/pdf/2011.14141v1.pdf





class LitModel(pl.LightningModule):
    def __init__(self,**cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder = timm.create_model(self.hparams['encoder'],features_only=True,pretrained=self.hparams['pretrained'])

        channels = self.encoder.feature_info.channels()
        self.clf = nn.Sequential(

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(channels[-1],self.hparams['num_classes'],1)
        )

        #self.loss_fn = FocalLossFromLogits(alpha=8.,gamma=2.)

        metrics = MetricCollection([               
                Accuracy(multiclass=False),
                Precision(multiclass=False,dist_sync_on_step=False),
                Recall(multiclass=False,dist_sync_on_step=False),
                F1(multiclass=False,dist_sync_on_step=False)])
        
        self.train_metrics = metrics.clone(prefix='{}/train_'.format(self.hparams['experiment_version']))
        self.valid_metrics = metrics.clone(prefix='{}/valid_'.format(self.hparams['experiment_version']))
        self.test_metrics = metrics.clone(prefix='{}/test_'.format(self.hparams['experiment_version']))

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.clf(x)
        x = x.squeeze(1).squeeze(2) 
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y,pos_weight=torch.tensor(1./.22))
        #loss = self.loss_fn(y_hat, y)

        with torch.no_grad():

            metrics = self.train_metrics(torch.sigmoid(y_hat),y.type(torch.int64))

        self.log_dict(metrics,on_step=True, on_epoch=True,prog_bar=True)
        self.log('{}/train_loss'.format(self.hparams['experiment_version']),loss)
        return loss



    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.binary_cross_entropy_with_logits(y_hat, y,pos_weight=torch.tensor(1./.22))
        #val_loss = self.loss_fn(y_hat, y)

        with torch.no_grad():
            metrics = self.valid_metrics(torch.sigmoid(y_hat),y.type(torch.int64))
            

        self.log_dict( metrics,on_step=True, on_epoch=True,prog_bar=True)
        self.log('{}/val_loss'.format(self.hparams['experiment_version']),val_loss)
        return val_loss



    def configure_optimizers(self):
     
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'])
        return optimizer


class BaseModel(pl.LightningModule):
    def __init__(self,**cfg):
        super().__init__()
        self.save_hyperparameters(cfg)


        
        
        
        #if self.hparams['add_laws']:
        #    self.laws = LawsLayer().float()
        #    for p in self.laws.parameters():
        #        p.requires_grad=False

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        metrics_dict = {}
        for k in self.train_metrics.keys():
            metrics_dict[k.split('/')[1]]=-1
        for k in self.valid_metrics.keys():
            metrics_dict[k.split('/')[1]]=-1
        self.logger.log_hyperparams(self.hparams, metrics_dict)
             
    def training_step(self, batch, batch_idx):
        sample = batch
        y_hat = self(sample['processed_image'])
        #loss = F.binary_cross_entropy_with_logits(y_hat['upsampled_logits'], sample['processed_mask'],weight=sample['weights'])
        loss = self.loss_fn(y_hat['upsampled_logits'], sample['processed_mask'])#,weight=sample['weights'])
        
        #if self.hparams['add_classification_head']:
        #    clf_loss = F.binary_cross_entropy_with_logits(y_hat['clf_logits'], sample['label'])
        #    loss = loss + clf_loss
        #    self.log('{}/train_clfloss'.format(self.hparams['experiment_version']),clf_loss,on_step=False,on_epoch=True,prog_bar=True)
        

        with torch.no_grad():

            metrics = self.train_metrics(y_hat['upsampled_logits'],sample['processed_mask'])

        self.log_dict(metrics,on_step=True, on_epoch=True,prog_bar=False)
        self.log('{}/train_loss'.format(self.hparams['experiment_version']),loss,on_step=False,on_epoch=True,prog_bar=True)
        return {'loss':loss,'y_hat':y_hat['upsampled_logits'].detach()}



    def validation_step(self, batch, batch_idx):
        sample = batch
        y_hat = self(sample['processed_image'])
        #val_loss = F.binary_cross_entropy_with_logits(y_hat['upsampled_logits'], sample['processed_mask'],weight=sample['weights'])
        val_loss = self.loss_fn(y_hat['upsampled_logits'], sample['processed_mask'])#,weight=sample['weights'])
        
        #if self.hparams['add_classification_head']:
        #    clf_loss = F.binary_cross_entropy_with_logits(y_hat['clf_logits'], sample['label'])
        #    val_loss = val_loss + clf_loss
        #    self.log('{}/val_clfloss'.format(self.hparams['experiment_version']),clf_loss,on_step=False,on_epoch=True,prog_bar=True)
              
        with torch.no_grad():
            metrics = self.valid_metrics(y_hat['upsampled_logits'],sample['processed_mask'])
            

        self.log_dict(metrics,on_step=True, on_epoch=True,prog_bar=False)
        self.log('{}/val_loss'.format(self.hparams['experiment_version']),val_loss,on_step=False,on_epoch=True,prog_bar=True)
        return {'val_loss':val_loss,'y_hat':y_hat['upsampled_logits'].detach()}



    def configure_optimizers(self):

        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'],weight_decay=self.hparams['l2_reg'])
        return optimizer

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

class BaseDepthModel(BaseModel):
    def __init__(self,**cfg):
        BaseModel.__init__(self,**cfg)

        self.loss_fn = ScaleInvariantLoss(**self.hparams)#FocalLossFromLogits(alpha=4,gamma=2,reduction='mean')

        
        metrics = MetricCollection([      
            MeanAbsoluteError(cfg=cfg),
            MeanSquaredError(cfg=cfg)])
        
        self.train_metrics = metrics.clone(prefix='{}/train_'.format(self.hparams['experiment_version']))
        self.valid_metrics = metrics.clone(prefix='{}/valid_'.format(self.hparams['experiment_version']))
        self.test_metrics = metrics.clone(prefix='{}/test_'.format(self.hparams['experiment_version']))
   
#=========== SEGMENTATION ===========#    
class BaseSegModel(BaseModel):
    def __init__(self,**cfg):
        BaseModel.__init__(self,**cfg)

        #self.encoder = timm.create_model(self.hparams['encoder'],
        #                                 features_only=True,
        #                                 pretrained=self.hparams['pretrained'],
        #                                 output_stride=self.hparams['output_stride'],
        #                                 in_chans=12 if self.hparams['add_laws'] else 3)

        metrics = MetricCollection([               
        Accuracy(multiclass=False),
        Precision(multiclass=False,dist_sync_on_step=False),
        Recall(multiclass=False,dist_sync_on_step=False),
        F1(multiclass=False,dist_sync_on_step=False),
        MyIoU(compute_on_step=True,dist_sync_on_step=False,threshold=0.5)])
        
        self.train_metrics = metrics.clone(prefix='{}/train_'.format(self.hparams['experiment_version']))
        self.valid_metrics = metrics.clone(prefix='{}/valid_'.format(self.hparams['experiment_version']))
        self.test_metrics = metrics.clone(prefix='{}/test_'.format(self.hparams['experiment_version']))


class FCNModel(BaseSegModel):
    def __init__(self,**cfg):
        BaseSegModel.__init__(self,**cfg)

        channels = self.encoder.feature_info.channels()
        self.segmentation_head = nn.Sequential(
            nn.Dropout2d(self.hparams['dropout_p']),
            nn.Conv2d(channels[-1],channels[-1]//2,1),
            nn.BatchNorm2d(channels[-1]//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.hparams['dropout_p']),
            nn.Conv2d(channels[-1]//2,self.hparams['num_classes'],1)
        )

        #self.loss_fn = FocalLossFromLogits(alpha=8.,gamma=2.)
        if self.hparams['add_classification_head']:
            self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(channels[-1],self.hparams['num_classes'],1)
        )



    def forward(self, x):
        if self.hparams['add_laws']:
            x = torch.cat([x,self.laws(x)],axis=1)
            
        y = self.encoder(x)[-1]
        
        clf_logits = None
        if self.hparams['add_classification_head']:
            clf_logits = self.classification_head(y).squeeze(1).squeeze(1).squeeze(1)
            
        y = self.segmentation_head(y)
        upsampled_logits = nn.functional.interpolate(y,size=x.shape[2:],mode='bilinear',align_corners=True)
        
        return {'upsampled_logits':upsampled_logits,'clf_logits': clf_logits}

import collections
from xlightning.custom_layers import *
from xlightning.custom_layers import _AtrousSpatialPyramidPoolingModule
import timm
from xlightning.losses import FocalLossFromLogits

class DoNothing(torch.nn.Module):
    def __init__(self):
        super(DoNothing, self).__init__()
        pass
    def forward(self, x):
        return x
  
class DeepLabV3plus_custom(BaseSegModel):
    def __init__(self,**cfg):
        BaseSegModel.__init__(self,**cfg)
    
        encoder_channels = self.encoder.feature_info.channels()
        self.aspp = _AtrousSpatialPyramidPoolingModule(in_dim=encoder_channels[-1], reduction_dim=256, output_stride=self.hparams['output_stride'], rates=[6, 12, 18])
        self.bot_aspp = nn.Sequential(collections.OrderedDict([('conv1',nn.Conv2d(1280, 256, kernel_size=1, bias=False))]))
        self.bot_fine = nn.Sequential(collections.OrderedDict([('conv1',nn.Conv2d(encoder_channels[1 if self.hparams['output_stride']==8 else 2], 48, kernel_size=1, bias=False))]))
        
        self.interpolate = torch.nn.functional.interpolate
        self.dropout2D_1 = nn.Dropout2d(self.hparams['dropout_p']) if isinstance(self.hparams['dropout_p'],float) else DoNothing()
        self.dropout2D_2 = nn.Dropout2d(self.hparams['dropout_p']) if isinstance(self.hparams['dropout_p'],float) else DoNothing()
        self.final_seg = nn.Sequential(
            collections.OrderedDict([
            
            ('conv1',nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False)),
            ('bn1',nn.BatchNorm2d(256)),
            ('act1',nn.ReLU(inplace=True)),
            ('dout1',nn.Dropout2d(.3) if isinstance(self.hparams['dropout_p'],float) else DoNothing()),
            ('conv2',nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ('bn2',nn.BatchNorm2d(256)),
            ('act2',nn.ReLU(inplace=True)),
            ('dout2',nn.Dropout2d(.2) if isinstance(self.hparams['dropout_p'],float) else DoNothing()),
            ('clf',nn.Conv2d(256, self.hparams['num_classes'], kernel_size=1, bias= False))]))

        self.lr = cfg['lr']
        
        if self.hparams['add_classification_head']:
            self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(encoder_channels [-1],self.hparams['num_classes'],1)
        )
        self.loss_fn= berHu(**cfg) if cfg['loss']=='berHu' else ScaleInvariantLoss(**cfg)


    
    def forward(self,x):
        
        x_size = x.size() 
        
        if self.hparams['add_laws']:
            x = torch.cat([x,self.laws(x)],axis=1)
        
        fmaps = self.encoder(x)
        
        clf_logits = None
        if self.hparams['add_classification_head']:
            clf_logits = self.classification_head(fmaps[-1]).squeeze(1).squeeze(1).squeeze(1)
        
        fmap_fine = self.dropout2D_2(fmaps[1 if self.hparams['output_stride']==8 else 2])

        fmap_coarse = self.aspp(self.dropout2D_1(fmaps[-1]))
        
        
        fmap_coarse = self.bot_aspp(fmap_coarse)
        
        fmap_coarse_up = self.interpolate(fmap_coarse, fmap_fine.size()[2:], mode='bilinear',align_corners=True)
        
        fmap_fine = self.bot_fine(fmap_fine)
        
        dec0 = torch.cat([fmap_coarse_up,fmap_fine], 1)
        
        logits = self.final_seg(dec0)
        
        upsampled_logits = self.interpolate(logits, x_size[2:], mode='bilinear',align_corners=True)
        
        
        
        return {'upsampled_logits':upsampled_logits,'clf_logits': clf_logits}


#=============MONOCULAR DEPTH===============#
#======= COARSE AND FINE NETS ======#


class FusedEncoder(torch.nn.Module):
    def __init__(self,in_chans=3,**cfg):
        super().__init__()
        
        self.encoder_rgb = timm.create_model(cfg['encoder'],
                                         features_only=True,
                                         pretrained=cfg['pretrained'],
                                         output_stride=cfg['output_stride'],
                                         in_chans=12 if cfg['add_laws'] else in_chans)
        
        self.encoder_depth = timm.create_model(cfg['encoder'],
                                         features_only=True,
                                         pretrained=cfg['pretrained'],
                                         output_stride=cfg['output_stride'],in_chans=1)

        self.channels = self.encoder_rgb.feature_info.channels()
    def forward(self,inputs):
        rgb,depth = inputs
        
        y_rgb = self.encoder_rgb(rgb)
        y_depth = self.encoder_depth(depth)
        
        features = [i+j for i,j in zip(y_rgb,y_depth)]

        return features
        
class DeepLabV3plusDepthFine(BaseDepthModel):
    def __init__(self,checkpoint=None,in_chans = 3,**cfg):
        BaseDepthModel.__init__(self,**cfg)

        self.coarse_net = DeepLabV3plus_Depthcustom.load_from_checkpoint(checkpoint) 
        self.coarse_net.eval()

        self.encoder = FusedEncoder(**self.hparams) 
        encoder_channels = self.encoder.channels
        self.aspp = _AtrousSpatialPyramidPoolingModule(in_dim=encoder_channels[-1], reduction_dim=256, output_stride=self.hparams['output_stride'], rates=[6, 12, 18] if self.hparams['output_stride']==16 else [12,24,36])
        self.bot_aspp = nn.Sequential(collections.OrderedDict([('conv1',nn.Conv2d(1280, 256, kernel_size=1, bias=False))]))
        self.bot_fine = nn.Sequential(collections.OrderedDict([('conv1',nn.Conv2d(encoder_channels[1 if self.hparams['output_stride']==8 else 2], 48, kernel_size=1, bias=False))]))
        
        self.interpolate = torch.nn.functional.interpolate
        self.dropout2D_1 = nn.Dropout2d(self.hparams['dropout_p']) if isinstance(self.hparams['dropout_p'],float) else DoNothing()
        #self.dropout2D_2 = nn.Dropout2d(self.hparams['dropout_p']) if isinstance(self.hparams['dropout_p'],float) else DoNothing()
        self.final_seg = nn.Sequential(
            collections.OrderedDict([
            
            ('conv1',nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False)),
            ('bn1',nn.BatchNorm2d(256)),
            ('act1',nn.ReLU(inplace=True)),
            #('dout1',nn.Dropout2d(.3) if isinstance(self.hparams['dropout_p'],float) else DoNothing()),
            ('conv2',nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ('bn2',nn.BatchNorm2d(256)),
            ('act2',nn.ReLU(inplace=True)),
            #('dout2',nn.Dropout2d(.2) if isinstance(self.hparams['dropout_p'],float) else DoNothing()),
            ('clf',nn.Conv2d(256, 1, kernel_size=1, bias= False))]))

        self.lr = cfg['lr']
        
        if self.hparams['add_classification_head']:
            self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(encoder_channels [-1],self.hparams['num_classes'],1)
        )

        self.loss_fn =  {'scale':ScaleInvariantLoss, 'berHu': berHu, 'l1':L1Loss,'scale_gradient':ScaleInvariantLossGradient}[self.hparams['loss']](**cfg)#


    
    def forward(self,x):
        
        x_size = x.size() 
        
        if self.hparams['add_laws']:
            x = torch.cat([x,self.laws(x)],axis=1)
        
        with torch.no_grad():
            z = self.coarse_net(x)['upsampled_logits']

        fmaps = self.encoder((x,z))
        
        clf_logits = None
        if self.hparams['add_classification_head']:
            clf_logits = self.classification_head(fmaps[-1]).squeeze(1).squeeze(1).squeeze(1)
        
        fmap_fine = fmaps[1 if self.hparams['output_stride']==8 else 2]

        fmap_coarse = self.aspp(self.dropout2D_1(fmaps[-1]))
        
        
        fmap_coarse = self.bot_aspp(fmap_coarse)
        
        fmap_coarse_up = self.interpolate(fmap_coarse, fmap_fine.size()[2:], mode='bilinear',align_corners=True)
        
        fmap_fine = self.bot_fine(fmap_fine)
        
        dec0 = torch.cat([fmap_coarse_up,fmap_fine], 1)
        
        logits = self.final_seg(dec0)
        
        upsampled_logits = self.interpolate(logits, x_size[2:], mode='bilinear',align_corners=True)
      
        return {'upsampled_logits':upsampled_logits,'clf_logits': clf_logits}

class DeepLabV3plus_Depthcustom(BaseDepthModel):
    def __init__(self,in_chans = 3,**cfg):
        BaseDepthModel.__init__(self,**cfg)

        self.encoder = timm.create_model(self.hparams['encoder'],
                                         features_only=True,
                                         pretrained=self.hparams['pretrained'],
                                         output_stride=self.hparams['output_stride'],
                                         in_chans=12 if self.hparams['add_laws'] else in_chans)   
        encoder_channels = self.encoder.feature_info.channels()
        self.aspp = _AtrousSpatialPyramidPoolingModule(in_dim=encoder_channels[-1], reduction_dim=256, output_stride=self.hparams['output_stride'], rates=[6, 12, 18] if self.hparams['output_stride']==16 else [12,24,36])
        self.bot_aspp = nn.Sequential(collections.OrderedDict([('conv1',nn.Conv2d(1280, 256, kernel_size=1, bias=False))]))
        self.bot_fine = nn.Sequential(collections.OrderedDict([('conv1',nn.Conv2d(encoder_channels[1 if self.hparams['output_stride']==8 else 2], 48, kernel_size=1, bias=False))]))
        
        self.interpolate = torch.nn.functional.interpolate
        self.dropout2D_1 = nn.Dropout2d(self.hparams['dropout_p']) if isinstance(self.hparams['dropout_p'],float) else DoNothing()
        #self.dropout2D_2 = nn.Dropout2d(self.hparams['dropout_p']) if isinstance(self.hparams['dropout_p'],float) else DoNothing()
        self.final_seg = nn.Sequential(
            collections.OrderedDict([
            
            ('conv1',nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False)),
            ('bn1',nn.BatchNorm2d(256)),
            ('act1',nn.ReLU(inplace=True)),
            #('dout1',nn.Dropout2d(.3) if isinstance(self.hparams['dropout_p'],float) else DoNothing()),
            ('conv2',nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ('bn2',nn.BatchNorm2d(256)),
            ('act2',nn.ReLU(inplace=True)),
            #('dout2',nn.Dropout2d(.2) if isinstance(self.hparams['dropout_p'],float) else DoNothing()),
            ('clf',nn.Conv2d(256, 1, kernel_size=1, bias= False))]))

        self.lr = self.hparams['lr']
        
        if self.hparams['add_classification_head']:
            self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(encoder_channels [-1],self.hparams['num_classes'],1)
        )

        self.loss_fn =  {'scale':ScaleInvariantLoss, 'berHu': berHu, 'l1':L1Loss,'scale_gradient':ScaleInvariantLossGradient}[self.hparams['loss']](**cfg)#
   
    def forward(self,x):
        
        x_size = x.size() 
        
        if self.hparams['add_laws']:
            x = torch.cat([x,self.laws(x)],axis=1)
        
        fmaps = self.encoder(x)
        
        clf_logits = None
        if self.hparams['add_classification_head']:
            clf_logits = self.classification_head(fmaps[-1]).squeeze(1).squeeze(1).squeeze(1)
        
        fmap_fine = fmaps[1 if self.hparams['output_stride']==8 else 2]

        fmap_coarse = self.aspp(self.dropout2D_1(fmaps[-1]))
        
        
        fmap_coarse = self.bot_aspp(fmap_coarse)
        
        fmap_coarse_up = self.interpolate(fmap_coarse, fmap_fine.size()[2:], mode='bilinear',align_corners=True)
        
        fmap_fine = self.bot_fine(fmap_fine)
        
        dec0 = torch.cat([fmap_coarse_up,fmap_fine], 1)
        
        logits = self.final_seg(dec0)
        
        upsampled_logits = self.interpolate(logits, x_size[2:], mode='bilinear',align_corners=True)
        
        
        
        return {'upsampled_logits':upsampled_logits,'clf_logits': clf_logits}

class ____FineDeepLabV3plus_Depthcustom(DeepLabV3plus_Depthcustom):
    def __init__(self,checkpoint,**cfg):
        DeepLabV3plus_Depthcustom.__init__(self,in_chans=1,**cfg)
        self.coarse_net = DeepLabV3plus_Depthcustom.load_from_checkpoint(checkpoint) if isinstance(checkpoint,str) else Generator(in_chans=1,**self.hparams)
    def forward(self, z):
        with torch.no_grad():
            x = self.coarse_net(z)['upsampled_logits']
  

        
        x_size = x.size() 
        
        if self.hparams['add_laws']:
            x = torch.cat([x,self.laws(x)],axis=1)
        
        fmaps = self.encoder(x)
        
        clf_logits = None
        if self.hparams['add_classification_head']:
            clf_logits = self.classification_head(fmaps[-1]).squeeze(1).squeeze(1).squeeze(1)
        
        fmap_fine = fmaps[1 if self.hparams['output_stride']==8 else 2]

        fmap_coarse = self.aspp(self.dropout2D_1(fmaps[-1]))
        
        
        fmap_coarse = self.bot_aspp(fmap_coarse)
        
        fmap_coarse_up = self.interpolate(fmap_coarse, fmap_fine.size()[2:], mode='bilinear',align_corners=True)
        
        fmap_fine = self.bot_fine(fmap_fine)
        
        dec0 = torch.cat([fmap_coarse_up,fmap_fine], 1)
        
        logits = self.final_seg(dec0)
        
        upsampled_logits = self.interpolate(logits, x_size[2:], mode='bilinear',align_corners=True)
        

        return {'upsampled_logits':upsampled_logits,'clf_logits': clf_logits}
#============ GAN ================#

class Generator(DeepLabV3plus_Depthcustom):
    def __init__(self,**cfg):
        DeepLabV3plus_Depthcustom.__init__(self,**cfg)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = timm.create_model('vgg16',features_only=True,pretrained=True, in_chans=1)
        self.sigmoid = torch.sigmoid
    def forward(self, input):
        y = self.encoder(input)[-1]
        return self.sigmoid(y)

class Discriminator_old(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 32
        nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.AvgPool2d(kernel_size=4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class GAN(pl.LightningModule):
    def __init__(
        self,
        validation_z,
        cfg,
        #latent_dim: int = 100,
        checkpoint=None,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters(cfg)

        # networks
        print(cfg)
 
        self.generator = Generator(**cfg)
        self.discriminator = Discriminator()

        self.validation_z = validation_z#torch.randn(8, self.hparams.latent_dim)

        #self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)['upsampled_logits']

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs=batch['processed_image']
        depthmaps = batch['processed_mask']

        # sample noise
        z = imgs
     

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)
            y_hat = self.generated_imgs 

            # log sampled images
            #sample_imgs = self.generated_imgs[:6]
            #sample_imgs  = (sample_imgs / sample_imgs .max())
            #grid = torchvision.utils.make_grid(sample_imgs)
            #self.logger.experiment.add_image("generated_images", grid, 0)



            # adversarial loss is binary cross-entropy
            D_out = self.discriminator(self(z)).view(-1,1)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(D_out.size(0), 1)
            valid = valid.type_as(imgs)
            
            g_loss = self.adversarial_loss( D_out, valid)

            l1_loss = self.generator.loss_fn(self.generated_imgs,depthmaps)

            tqdm_dict = {"g_loss": g_loss,"l1_loss":l1_loss}
            output = OrderedDict({"loss": g_loss+l1_loss,"y_hat":y_hat, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples


            D_out = self.discriminator(depthmaps).view(-1,1)
            # how well can it label as real?
            valid = torch.ones(D_out.size(0), 1)
            valid = valid.type_as(D_out)

            real_loss = self.adversarial_loss(D_out, valid)

            # how well can it label as fake?
            fake = torch.zeros(D_out.size(0), 1)
            fake = fake.type_as(D_out)

            y_hat= self(z).detach()

            fake_loss = self.adversarial_loss(self.discriminator(y_hat).view(-1,1), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict,"y_hat":y_hat.detach(), "log": tqdm_dict})
            return output

    def validation_step(self, batch, batch_idx):
        sample = batch
        y_hat = self(sample['processed_image'])
        #val_loss = F.binary_cross_entropy_with_logits(y_hat['upsampled_logits'], sample['processed_mask'],weight=sample['weights'])
        val_loss = self.generator.loss_fn(y_hat, sample['processed_mask'])#,weight=sample['weights'])
        
        #if self.hparams['add_classification_head']:
        #    clf_loss = F.binary_cross_entropy_with_logits(y_hat['clf_logits'], sample['label'])
        #    val_loss = val_loss + clf_loss
        #    self.log('{}/val_clfloss'.format(self.hparams['experiment_version']),clf_loss,on_step=False,on_epoch=True,prog_bar=True)
              
        with torch.no_grad():
            metrics = self.generator.valid_metrics(y_hat,sample['processed_mask'])
            

        self.log_dict(metrics,on_step=True, on_epoch=True,prog_bar=False)
        self.log('{}/val_loss'.format(self.hparams['experiment_version']),val_loss,on_step=False,on_epoch=True,prog_bar=True)
        return {'val_loss':val_loss,'y_hat':y_hat.detach()}

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.final_seg.conv1.weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

class FineGAN(GAN):
    def __init__(self,
            validation_z,
            #latent_dim: int = 100,
            checkpoint=None,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,**cfg):

        GAN.__init__(self,        
        validation_z,
        #latent_dim: int = 100,
        checkpoint=None,
        lr= lr,
        b1= b1,
        b2 = b2,**cfg)

        self.coarse_net = Generator.load_from_checkpoint(checkpoint) if isinstance(checkpoint,str) else Generator(**self.hparams)
    def forward(self, z):
        with torch.no_grad():
            z = self.coarse_net(z)['upsampled_logits']
        return self.generator(z)['upsampled_logits']

#============ END GAN ================#

#============ DPT ================#

from xlightning.losses import berHu, ScaleInvariantLoss ,L1Loss, ScaleInvariantLossGradient
from xlightning.dpt.models import DPTSegmentationModel, DPTDepthModel

class Scale(nn.Module):
    def __init__(self,**cfg):
        super.__init__(self)
        pass

    def forward(self,x):

        return x#/1000.


class DPTmodel(BaseDepthModel):
    def __init__(self,**cfg):
        BaseDepthModel.__init__(self,**cfg)
        

        default_models = {
        "midas_v21": "xlightning/dpt/weights/midas_v21-f6b98070.pt",
        "dpt_large": "xlightning/dpt/weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "xlightning/dpt/weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "xlightning/dpt/weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "xlightning/dpt/weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }
    
        if self.hparams['dpt_model'] == 'dpt_hybrid_nyu':
            self.dpt = DPTDepthModel(
                path=default_models[self.hparams['dpt_model']],
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.scale = nn.Identity()

          # load network
        elif self.hparams['dpt_model'] == "dpt_large":  # DPT-Large
            self.dpt= DPTDepthModel(
                path=default_models[self.hparams['dpt_model']],
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            self.scale = nn.Identity()
        
        elif self.hparams['dpt_model'] == "dpt_large":  # DPT-Large
            self.dpt= DPTDepthModel(
                path=default_models[self.hparams['dpt_model']],
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            self.scale = nn.Identity()

        elif self.hparams['dpt_model'] == "dpt_hybrid":
            
            self.dpt = DPTDepthModel(
                        path=default_models[self.hparams['dpt_model']],
                        backbone="vitb_rn50_384",
                        non_negative=False,
                        invert=False,
                        enable_attention_hooks=False,
                    )
            self.scale =nn.Identity()

        elif self.hparams['dpt_model'] == "dpt_hybrid_scratch":
            
            self.dpt = DPTDepthModel(
                        path=None,
                        backbone="vitb_rn50_384",
                        non_negative=False,
                        enable_attention_hooks=False,
                    )
            self.scale = nn.Identity()
        else:
            raise

 
        """
        self.final_reg = nn.Sequential(
            collections.OrderedDict([
            

            ('bn1',nn.BatchNorm2d(150)),
            ('act1',nn.ReLU(inplace=True)),
            ('conv2',nn.Conv2d(150, 1, kernel_size=1, padding=0, bias=False)),
            ]))
        """

        self.lr = cfg['lr']

        self.loss_fn =  {'scale':ScaleInvariantLoss, 'berHu': berHu, 'l1':L1Loss,'scale_gradient':ScaleInvariantLossGradient}[self.hparams['loss']](**cfg)#
    

    def forward(self,x):
      
        upsampled_logits = self.dpt(x).unsqueeze(1)

        upsampled_logits = self.scale(upsampled_logits)

        #t = torch.median(upsampled_logits)
        #s = torch.mean(torch.abs(upsampled_logits - t))
        #upsampled_logits = (upsampled_logits-t)/s

        #upsampled_logits = self.norm(upsampled_logits.unsqueeze(1)).squeeze(1)

        return {'upsampled_logits':upsampled_logits,'clf_logits': None}


    def configure_optimizers(self):
        parameters = [
                {'params': self.dpt.pretrained.parameters(),'lr':self.hparams['lr']*.1},
                {'params': self.dpt.scratch.parameters(), 'lr': self.hparams['lr']},
                
            ]
        print(parameters)
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
        

#============ END DPT ================#

# ===> Contrastive Learning
import torchvision
class SimCLR(pl.LightningModule):
    def __init__(self, **cfg):
        #hidden_dim, lr, temperature, weight_decay, max_epochs=500
        super().__init__()
        self.save_hyperparameters(cfg)
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = torchvision.models.resnet50(
            pretrained=False, num_classes=4 * self.hparams.hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.hparams.hidden_dim, self.hparams.hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch,batch_idx, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log('{}/{}_loss'.format(self.logger.version,mode), nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        if batch_idx%100==0:
            with torch.no_grad():
                for tt in range(len(batch[0])):
                
                    self.logger.experiment.add_images(tag='{}/{}_{}-posimg'.format(self.logger.version,mode,tt),img_tensor=0.5*batch[0][tt]+.5,global_step=self.global_step,dataformats='NCHW')
                    #self.logger.experiment.add_images(tag='{}/{}_negimg'.format(self.logger.version,mode),img_tensor=imgs[~pos_mask],global_step=self.global_step,dataformats='NCHW')

        self.log('{}/{}_acc_top1'.format(self.logger.version,mode), (sim_argsort == 0).float().mean())
        self.log('{}/{}_acc_top5'.format(self.logger.version,mode), (sim_argsort < 5).float().mean())
        self.log('{}/{}_acc_mean_pos'.format(self.logger.version,mode), 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, batch_idx,mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, batch_idx,mode="val")

    def forward(self, x):
        return self.convnet(x)
    def __call__(self,x):
        return self.forward(x)  



 
#===================

#import vision.references.detection.utils as utils

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
from torchvision.models.detection.rpn import AnchorGenerator


from collections import OrderedDict
import torch.nn as nn

from torchvision.models.detection import MaskRCNN

#==> https://akashprakas.github.io/My-blog/jupyter/2020/12/19/Hacking_fasterRcnn.html#Custom-Backbone-with-FPN
#==> https://medium.com/jumio/object-detection-tutorial-with-torchvision-82b8f269f6ff
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork

from torchvision.models.detection import MaskRCNN



def get_model_instance_segmentation(num_classes):
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,progress=False)
                                                            
    #fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Define RPN 25.        , 240.66666667, 456.33333333, 672.  # 50,100,400,640+128
    #240, 456, 672,640+128
    anchor_generator = AnchorGenerator(sizes=tuple([( 240, 456, 672,640+128) for _ in range(5)]), # let num of tuple equal to num of feature maps
                                    aspect_ratios=tuple([(.5, 1.0,2) for _ in range(5)])) # ref: https://github.com/pytorch/vision/issues/978

                                                            
    rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    model.rpn = RegionProposalNetwork(
        anchor_generator= anchor_generator, head= rpn_head,
        fg_iou_thresh= 0.5, bg_iou_thresh=0.3,
        batch_size_per_image=32, # use fewer proposals
        positive_fraction = 0.22,
        pre_nms_top_n=dict(training=200, testing=100),#pre_nms_top_n=dict(training=200, testing=100),
        post_nms_top_n=dict(training=160, testing=80),#post_nms_top_n=dict(training=160, testing=80),
        nms_thresh = 0.7
    )
    
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    #model.roi_heads.mask_predictor.mask_fcn_logits = 
    return model
#==> https://forums.pytorchlightning.ai/t/coco-metrics-in-pytorch-lightning/935
class LitMaskRCNN(BaseSegModel):
        def __init__(self,**cfg):
            BaseSegModel.__init__(self,**cfg)
            
            
            self.model = get_model_instance_segmentation(num_classes=self.hparams['num_classes'])

        def forward(self,x):
            return self.model(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            assert len(y[0]['boxes'].shape)==4
            assert len(x[0].shape)==4
            loss_dict = self.model(x,y)
            losses = sum(loss*self.hparams[key] for key,loss in loss_dict.items())
            
            loss_dict = {'{}/{}'.format(self.hparams['experiment_version'],k):v for k,v in loss_dict.items()}
            self.log_dict(loss_dict,on_step=False, on_epoch=True,prog_bar=False)
            self.log('{}/train_loss'.format(self.hparams['experiment_version']),losses,on_step=False,on_epoch=True,prog_bar=True)
            return {'loss':losses}

        def validation_step(self, batch, batch_idx):
            x, y = batch

            y_hat = self.model(x)
            
            predicted_mask = []
            for i in y_hat:
                keep = i['scores']>.5
                if torch.sum(keep)>0:
                    m = i['masks'][keep]
                    if len(m.shape)==2:
                        m = m.unsqueeze(0)
                    predicted_mask.append(m)
                else:
                    predicted_mask.append(torch.zeros_like(x[0][0].unsqueeze(0)+.01))
                    
            predicted_masks = torch.max(torch.vstack(predicted_mask),axis=0).values.unsqueeze(0)
            if len(predicted_masks.shape)==3:
                predicted_masks=predicted_masks.unsqueeze(0)
            
            
            reference_masks = torch.vstack([torch.max(i['masks'],axis=0).values for i in y]).unsqueeze(0).unsqueeze(0)

            metrics = self.valid_metrics(predicted_masks,reference_masks.type(torch.int64))

            self.log_dict(metrics,on_step=False, on_epoch=True,prog_bar=False)

            return {'predicted_targets':y_hat ,'predicted_masks':predicted_masks}
            
            

        def configure_optimizers(self):
            # construct an optimizer
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=self.hparams['lr'],momentum=self.hparams['sgd_momentum'],
                                        weight_decay=self.hparams['l2_reg'])
            # and a learning rate scheduler
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=3,
                                                        gamma=0.1)

    
            
            return [optimizer],[lr_scheduler]

 
 
 