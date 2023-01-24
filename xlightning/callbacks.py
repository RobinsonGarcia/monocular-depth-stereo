import tkinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint, BaseFinetuning
from pytorch_lightning.callbacks import Callback
#from pytorch_lightning.callbacks import GPUStatsMonitor
import torch.nn as nn
from torchvision import transforms  

#== > https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/finetuning.html     
from torch.optim import Optimizer
import numpy as np
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union
def _cumulative_optimizer_frequencies(frequencies: Tuple[int]) -> np.ndarray:
    return np.cumsum(frequencies)

def _get_active_optimizers(
    optimizers: List[Optimizer], frequencies: List[int], batch_idx: Optional[int] = None
) -> List[Tuple[int, Optimizer]]:
    """Returns the currently active optimizers. When multiple optimizers are used with different frequencies, only
    one of the optimizers is active at a time.
    Returns:
        A list of tuples (opt_idx, optimizer) of currently active optimizers.
    """
    if not frequencies:
        # call training_step once per optimizer
        return list(enumerate(optimizers))

    freq_cumsum = _cumulative_optimizer_frequencies(tuple(frequencies))
    optimizers_loop_length = freq_cumsum[-1]
    current_place_in_loop = batch_idx % optimizers_loop_length

    # find optimizer index by looking for the first {item > current_place} in the cumsum list
    opt_idx = np.searchsorted(freq_cumsum, current_place_in_loop, side="right")
    return [(opt_idx, optimizers[opt_idx])]

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    
    def __init__(self, unfreeze_at_epoch):
        self._internal_optimizer_metadata: Dict[int, List[Dict[str, Any]]] = {}
        self._restarting=False
        self._unfreeze_at_epoch = unfreeze_at_epoch
        
    def freeze_before_training(self, pl_module):
         # freeze any module you want
         # Here, we are freezing `feature_extractor`
         self.freeze(pl_module.dpt.pretrained)
 
         
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            """Called when the epoch begins."""

            for opt_idx, optimizer in _get_active_optimizers(trainer.optimizers, trainer.optimizer_frequencies):
                num_param_groups = len(optimizer.param_groups)
                self.finetune_function(pl_module, trainer.current_epoch, optimizer, opt_idx)
                current_param_groups = optimizer.param_groups
                
                self._store(pl_module, opt_idx, num_param_groups, current_param_groups)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
         # When `current_epoch` is 10, feature_extractor will start training.
         
         
         if current_epoch == self._unfreeze_at_epoch:
          
             self.unfreeze_and_add_param_group(
                 modules=pl_module.dpt.pretrained,
                 optimizer=optimizer,
                 train_bn=True,
             )

#https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch
import pytorch_lightning as pl
class InputMonitor(pl.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        #if (batch_idx + 1) % trainer.log_every_n_steps == 0:
        if (batch_idx ==1) or (batch_idx ==10):
            x, y = batch
            if len(y)!=1:
                y=y[0]
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)



import torch.nn as nn
import math
import numpy as np
# ==> https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/03-initialization-and-optimization.html
class WeightInitializationCallback(pl.Callback):
    def __init__(self,**cfg):
        super().__init__()
        self.cfg = cfg
        pass
    def freeze_encoder_bn(self,encoder):
        for name,param in encoder.named_parameters():
            if 'bn' in name:
                for n in self.cfg['init_layers']:
                    if n in name:
                        continue
                    else:
                        param.requires_grad=False

    def kaiming_init(self,name,param):
        
            
        if 'bn' in name:
            if name.endswith('weight'):
                if len(param.data.shape)==1:
                    d2 = param.data.shape[0]
                else:
                    d2 = param.data.shape[1]
                
                param.data.normal_(0,1 / math.sqrt(2) / math.sqrt(d2))
            elif name.endswith('.bias'):
                param.data.fill_(0.)
        elif name.endswith(".bias"):
            param.data.fill_(0.)
        elif name.endswith('.weight'):

                #param.requires_grad=False
            if len(param.data.shape)==1:
                d2 = param.data.shape[0]
            else:
                d2 = param.data.shape[1]
            
            param.data.normal_(0, 1 / math.sqrt(d2))
        else:
            print('NOT INITIALIZED: {}'.format(name))
            


    def on_fit_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if name.startswith('encoder'):
                for l in self.cfg['init_layers']:
                    if l in name:
                        self.kaiming_init(name,param)
                    else:
                        continue
                continue
            else:
                self.kaiming_init(name,param)
        #if self.cfg['freeze_encoder_bn']:self.freeze_encoder_bn(pl_module.encoder)
        pass
        

class LogMetrics(pl.Callback):
    def on_train_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pass

    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pass

    def on_train_epoch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pass

    def on_validation_epoch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pass

"""
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# ==> https://github.com/jacobgil/pytorch-grad-camx
class LogCAM(pl.Callback):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.inv_transform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        

    def log_GradCam(self,*args,**kwargs):
        if  kwargs['batch_idx']%10==0:
            x, y = kwargs['batch']
            target_layers = [kwargs['pl_module'].encoder.layer4[-1]]
            
            
            with GradCAM(model= kwargs['pl_module'], target_layers=target_layers, use_cuda=False) as cam:

                grayscale_cam = cam(input_tensor=x, target_category=None)
                
            print(grayscale_cam.shape)

            rgb_img = self.inv_transform(x).detach().cpu().numpy()
            
            def colorize(y):
                cmap =  plt.get_cmap('jet')
                images = torch.stack([torch.tensor(cmap(i)[:,:,:-1]) for i in y[:,0]]).permute(0,3,1,2)
                            
                return images
                    
            color_cam = colorize(grayscale_cam[:,np.newaxis,:,:] ).detach().cpu().numpy()
            color_cam = 0.5*rgb_img + 0.5*color_cam
            
            kwargs['pl_module'].logger.experiment.add_images('{}/GradCAM{}_images'.format(kwargs['pl_module'].hparams['experiment_version'],kwargs['mode']),rgb_img,global_step=kwargs['pl_module'].global_step)
            kwargs['pl_module'].logger.experiment.add_images('{}/GradCAM{}_cam'.format(kwargs['pl_module'].hparams['experiment_version'],kwargs['mode']),color_cam,global_step=kwargs['pl_module'].global_step)
            
            
            pass

    
    def log_Cam(self,*args,**kwargs):
        if  kwargs['batch_idx']==10:
            x, y = kwargs['batch']
            with torch.no_grad():
                features = kwargs['pl_module'].encoder(x)
                cam =  kwargs['pl_module'].clf[1](features[-1])
                
                cam = nn.functional.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=True).detach()
            
            
                def colorize(y):
                    cmap =  plt.get_cmap('jet')
                    images = torch.stack([torch.tensor(cmap(i.cpu())[:,:,:-1]) for i in y[:,0]]).permute(0,3,1,2)
                        
                    return images
                
                cam = colorize(cam)
        
                
                x = self.inv_transform(x).detach().cpu()
                cam = 0.5*x + 0.5*cam
            kwargs['pl_module'].logger.experiment.add_images('{}/CAM_{}_images'.format(kwargs['pl_module'].hparams['experiment_version'],kwargs['mode']),x,global_step=kwargs['pl_module'].global_step)
            kwargs['pl_module'].logger.experiment.add_images('{}/CAM_{}_cam'.format(kwargs['pl_module'].hparams['experiment_version'],kwargs['mode']),cam,global_step=kwargs['pl_module'].global_step)
            
    def on_train_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.log_Cam(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx, unused=0,mode='train')
        #self.log_GradCam(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx, unused=0,mode='train')
        
        pass

    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.log_Cam(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx, unused=0,mode='valid')
        pass

    def _on_train_epoch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):

        pass

    def _on_validation_epoch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pass
"""  
import torch
class CheckBatchGradient(pl.Callback):
    
    
    def on_train_start(self, trainer, model):
        n = 0

        example_input = torch.rand(4,3,512,512)*torch.tensor(2) - torch.tensor(1)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")

class DepthLog(pl.Callback):
    def __init__(self,**cfg):
        self.cfg = cfg
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.inv_norm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        self.train_losses=[]
        self.valid_losses=[]
        self.epochs=[]
        pass
    def on_train_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if (batch_idx==1)|(batch_idx==5):

            sample = batch
     
            y_hat = outputs['y_hat'] if isinstance(outputs,dict) else outputs[0]['y_hat']
            global_step=trainer.global_step
            #try:
            #    bn_params = (pl_module.norm.running_var,pl_module.norm.running_mean)
            #except:
            #    bn_params=(None,None)

            self.tb_log(x=sample['processed_image'],y=sample['processed_mask'],y_hat=y_hat,ix=batch_idx,
            logger=pl_module.logger,hparams=pl_module.hparams,global_step=global_step,mode='train')

    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        #if (batch_idx==1)|(batch_idx==5):
        if batch_idx==0:
            sample = batch

            #if self.cfg['return_faces']:
            #    y, y_face = y
            #    y_face = y_face
            y_hat = outputs['y_hat'] if isinstance(outputs,dict) else outputs[0]['y_hat']
            global_step=trainer.global_step
            #try:
            #    bn_params = (pl_module.norm.running_var,pl_module.norm.running_mean)
            #except:
            #    bn_params=(None,None)
            self.tb_log(x=sample['processed_image'],y=sample['processed_mask'],y_hat=y_hat,ix=batch_idx,
            logger=pl_module.logger,hparams=pl_module.hparams,global_step=global_step,mode='val')





class BaseLogImagesCallback(Callback):
    def __init__(self,**cfg):
        self.cfg = cfg
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.inv_norm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        
    def unscale(self,inv_depth):
        with torch.no_grad():
            depth = self.cfg['scale'] * inv_depth + self.cfg['shift']
            depth[depth>6]=6
            depth = 1.0 / (depth+1e-8)
  
            return depth
    def on_train_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        #if (batch_idx==1)|(batch_idx==5):
        print(torch.mean(pl_module.norm_layer.weight),torch.mean(pl_module.norm_layer.bias))
        """
        if batch_idx==0:
            sample = batch

            #if self.cfg['return_faces']:
            #    y, y_face = y
            #    y_face = y_face
     
            y_hat = outputs['y_hat'] if isinstance(outputs,dict) else outputs[0]['y_hat']
            global_step=trainer.global_step
            #try:
            #    bn_params = (pl_module.norm.running_var,pl_module.norm.running_mean)
            #except:
            #    bn_params=(None,None)

            self.tb_log(x=sample['processed_image'],y=sample['processed_mask'],y_hat=y_hat,ix=batch_idx,
            logger=pl_module.logger,hparams=pl_module.hparams,global_step=global_step,mode='train',log=pl_module.log,log_pcd=False)
        """
        
    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        #if (batch_idx==1)|(batch_idx==5):
        if batch_idx%5 == 0:
            sample = batch

            #if self.cfg['return_faces']:
            #    y, y_face = y
            #    y_face = y_face
            y_hat = outputs['y_hat'] if isinstance(outputs,dict) else outputs[0]['y_hat']
            global_step=trainer.global_step
            #try:
            #    bn_params = (pl_module.norm.running_var,pl_module.norm.running_mean)
            #except:
            #    bn_params=(None,None)
            self.tb_log(x=sample['processed_image'],y=sample['processed_mask'],filename=sample['filename'],y_hat=y_hat,ix=batch_idx,
            logger=pl_module.logger,hparams=pl_module.hparams,global_step=global_step,mode='val',log=pl_module.log,log_pcd=True)

class CORRLogImagesCallback(BaseLogImagesCallback):
    def tb_log(self,x,y,y_hat,ix,logger,global_step,hparams,mode='train'):
        with torch.no_grad():

            logger.experiment.add_histogram(tag='{}/{}_activations'.format(hparams['experiment_version'],mode),values=y_hat,global_step=global_step)


            imgs = (255.*self.inv_norm(x)).float()

            predictions = torch.sigmoid(y_hat)

            logger.experiment.add_pr_curve('{}/{}_pr_curve'.format(hparams['experiment_version'],mode), y, predictions, global_step)

            def plot(mask,imgs):
                m = torch.zeros_like(imgs)
                m[:,0,:,:] = mask[:,0,:,:]*255.
                imgs = [(0.7*img + 0.3*mask).type(torch.uint8) for img,mask in zip(imgs,m)]
                grid = torch.stack(imgs)
                return grid

            #grid = make_grid(imgs)
            masks = (predictions>.5).float()
            logger.experiment.add_images(tag='{}/{}_gt_samples'.format(hparams['experiment_version'],mode),img_tensor=plot(y,imgs),global_step=global_step)
            logger.experiment.add_images(tag='{}/{}_pr_samples'.format(hparams['experiment_version'],mode),img_tensor=plot(masks,imgs),global_step=global_step)
            

            pass
from xlightning.depthmap import depthmap_to_3D
#div = lambda a,b: np.divide(a, b, out=np.zeros_like(a), where=b!=0)

import open3d as o3d
import os
def save_pointcloud(xyz,rgb,logger,t,mode):
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    B = xyz.shape[0]
    for i in range(B):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[i].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb[i].cpu().numpy()/255.)
        
        #path = os.path.join(logger._save_dir,logger._name,str(logger._version),'cloud_{}_{}.ply'.format(mode,i))
        path = os.path.join(logger._save_dir,'cloud_{}_{}_{}.ply'.format(mode,i,t))
        
        o3d.io.write_point_cloud(path, pcd)
    
    pass


import wandb
import pdb
class DEPTHLogImagesCallback(BaseLogImagesCallback):
    def tb_log(self,x,y,filename,y_hat,ix,logger,global_step,hparams,mode='train',log=None,log_pcd=False,use_wandb=False):
        with torch.no_grad():

            imgs = (self.inv_norm(x)).float()

                
                
            #mask = (y_hat < self.cfg['min_dist']) | (y_hat > self.cfg['max_dist'])
            #y[mask]=0.0
            #y_hat[mask]=0.0

            #y = torch.clamp(y, min=self.cfg['min_dist'], max=self.cfg['max_dist'])
            #y_hat = torch.clamp(y_hat, min=self.cfg['min_dist'], max=self.cfg['max_dist'])

            target = y
            
            B,C,H,W = x.shape

            if log_pcd:
                xyz_target_t = depthmap_to_3D(y,self.cfg['adjust_depth']).permute(0,2,3,1).reshape(B,H*W,3)
                
                y_t = y.permute(0,2,3,1).reshape(B,H*W) 
                mask = (y_t < self.cfg['min_dist']) | (y_t > self.cfg['max_dist'])
                #mask = mask.reshape(B,H*W)

                mask1 = torch.mean(xyz_target_t,axis=-1) == 0.0
                rgb_t_1 = (255*imgs.permute(0,2,3,1).reshape(B,H*W,3))
                #rgb_t_1=rgb_t_1[mask]
                #rgb_t_1[mask]=torch.tensor([0,0,0]).type_as(rgb_t_1)

                xyz_pred_t = depthmap_to_3D(y_hat).permute(0,2,3,1).reshape(B,H*W,3)
                #mask2 = torch.mean(xyz_pred_t,axis=-1) == 0.0
                rgb_t_2 = (255*imgs.permute(0,2,3,1).reshape(B,H*W,3))
                #rgb_t_2= rgb_t_2[mask]#torch.tensor([0,0,0]).type_as(rgb_t_1)
                
                #print(xyz_target_t.shape,xyz_pred_t.shape,rgb_t_1.shape,rgb_t_2.shape)

                #xyz_target_t = xyz_target_t[mask,:]
                #xyz_pred_t = xyz_pred_t[mask,:]

                #print(xyz_target_t.shape,xyz_pred_t.shape,rgb_t_1.shape,rgb_t_2.shape)
                #pdb.set_trace()
                point_cloud = torch.hstack([xyz_target_t[0],rgb_t_1[0]])
                #print(point_cloud.shape)
                
                point_cloud = point_cloud[~mask[0],:].reshape(-1,6)
                #print(point_cloud.shape)

                #point_cloud = torch.hstack([xyz_target_t[0],rgb_t_1[0]])

                if use_wandb:
                    logger.experiment.log({
                        "{}_pcd_target_{}".format(filename).format(mode):wandb.Object3D(point_cloud.cpu().numpy())
                    })

            
                #log({"point_cloud_target": wandb.Object3D(point_cloud.cpu().numpy())})
                #print(point_cloud.shape)
                point_cloud = torch.hstack([xyz_pred_t[0],rgb_t_2[0]])

                point_cloud = point_cloud[~mask[0],:].reshape(-1,6)
                #print(point_cloud.shape)
                #point_cloud = torch.hstack([xyz_pred_t[0],rgb_t_2[0]])
                if use_wandb:
                    logger.experiment.log({
                        "{}_pcd_pred_{}".format(mode,filename):wandb.Object3D(point_cloud.cpu().numpy())
                    })
            #log({"point_cloud_pred": wandb.Object3D(point_cloud.cpu().numpy())})
            """"

            point_scene = wandb.Object3D({
                "type": "lidar/beta",
                "points":xyz_pred_t,
                "color":rgb_t_2,
            })
            wandb.log({"point_scene": point_scene})

            point_scene = wandb.Object3D({
                "type": "lidar/beta",
                "points":xyz_target_t,
                "color":rgb_t_1,
            })
            wandb.log({"point_scene_target": point_scene})
            """

            
            #logger.experiment.add_mesh('{}/{}_gt'.format(hparams['experiment_version'],mode),vertices=xyz_target_t,colors=rgb_t_1)
            #logger.experiment.add_mesh('{}/{}_pred'.format(hparams['experiment_version'],mode),vertices=xyz_pred_t,colors=rgb_t_2)
            #save_pointcloud(xyz_target_t,rgb_t_1,logger,mode)
            #save_pointcloud(xyz_pred_t,rgb_t_2,logger,mode)

            #logger.log({"predictions": wandb.Histogram(y_hat)})
            #logger.log({"gt": wandb.Histogram(y)})
            
            
            #logger.experiment.add_histogram(tag='{}/{}_scaled_pred'.format(hparams['experiment_version'],mode),values=y_hat,global_step=global_step)
            #logger.experiment.add_histogram(tag='{}/{}_scaled_target'.format(hparams['experiment_version'],mode),values=y,global_step=global_step)


            #logger.experiment.add_histogram(tag='{}/{}_unscaled_pred'.format(hparams['experiment_version'],mode),values=y_hat,global_step=global_step)
            #logger.experiment.add_histogram(tag='{}/{}_unscaled_target'.format(hparams['experiment_version'],mode),values=y,global_step=global_step)

        
            B = y.shape[0]
            mask = (y < self.cfg['min_dist']) | (y > self.cfg['max_dist'])
            y_hat[mask]=0.0
            y[mask]=0.0
   
            y_max = torch.max(y.reshape(B,-1),dim=1).values.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            #y_max = self.cfg['max_dist']#y_max.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            
            y_hat = (y_hat+torch.tensor(.001))/(y_max+torch.tensor(.001))
            #y_hat = y_hat/y_max

            y = (y+torch.tensor(.001))/(y_max+torch.tensor(.001))

            #y = y/y_max

            def colorize(y):
                cmap =  plt.get_cmap(hparams['cmap'])
                images = torch.stack([torch.tensor(cmap(i.cpu())[:,:,:-1]) for i in y[:,0]]).permute(0,3,1,2)  
                return images

            abs_error = torch.abs(y - y_hat)
            abs_error = colorize(abs_error)
            y = colorize(y)
            y_hat = colorize(y_hat)



            # Option 2: log images and predictions as a W&B Table
            n = 4
            columns = ['image', 'ground_truth', 'prediction','abs_error']
            data = [[wandb.Image(x_i), wandb.Image(y_i), wandb.Image(y_pred),wandb.Image(y_err)] for x_i, y_i, y_pred,y_err in list(zip(imgs[:n], y[:n], y_hat[:n],abs_error[:n]))]
            logger.log_table(
                key='Prediction table, batch: {}'.format(ix),
                columns=columns,
                data=data)

            



            #logger.experiment.add_images(tag='{}/{}_img'.format(hparams['experiment_version'],mode),img_tensor=imgs,global_step=global_step,dataformats='NCHW')
            #logger.experiment.add_images(tag='{}/{}_y_gt'.format(hparams['experiment_version'],mode),img_tensor=y,global_step=global_step,dataformats='NCHW')
            #logger.experiment.add_images(tag='{}/{}_y_pr'.format(hparams['experiment_version'],mode),img_tensor=y_hat,global_step=global_step,dataformats='NCHW')
            #logger.experiment.add_images(tag='{}/{}_error'.format(hparams['experiment_version'],mode),img_tensor=abs_error,global_step=global_step,dataformats='NCHW')

        pass


from PIL import Image





def load_callbacks(**cfg):
    monitor = cfg['monitor']
    return [
        #LogCAM(),
        #FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=cfg['unfreeze_at_epoch']),
        #DEPTHLogImagesCallback(**cfg) if cfg['DEPTHNET'] else CORRLogImagesCallback(**cfg),
        #InputMonitor(), 
        #CheckBatchGradient(),   
        DEPTHLogImagesCallback(**cfg) if cfg['depthnet'] else CORRLogImagesCallback(**cfg),
        #CORRLogImagesCallback(**cfg),     
        EarlyStopping(monitor=cfg['monitor'], patience=10,mode='min',check_on_train_epoch_end=False),                   
        ModelCheckpoint(dirpath=cfg['checkpoint_path'],auto_insert_metric_name =False,
        mode='min',monitor=cfg['monitor'],save_top_k=2),
        #LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        #WeightInitializationCallback(**cfg),
        #GPUStatsMonitor()
        #CheckBatchGradient()
        ]                  
