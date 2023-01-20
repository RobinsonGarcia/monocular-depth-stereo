import matplotlib.pyplot as plt
import pandas as pd
import xlightning.models as xlm
import xlightning.data as xld
from xlightning.data.datamodules import EigenDataModule3
import pytorch_lightning as pl
from xlightning.callbacks import load_callbacks
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from argparse import ArgumentParser
from xlightning.utils import override_cfg
from xlightning.models.depth import gans_wrapper
import torch
import json
import os

from xlightning.models.depth.pl_deeplabv3plus import *
from xlightning.models.depth.pl_dpt import *
from xlightning.models.depth.pl_dpt2 import *

from xlightning.models.depth.pl_glp import *
from xlightning.models.segmentation.pl_deeplabv3plus import *
from xlightning.models.depth.pl_newcrf import NewCRF
from xlightning.models.depth.pl_adabins import AdaBins
from xlightning.models.depth.pl_bts import BTS


from xlightning.losses import L1Loss

from xlightning.losses import L1Loss


class GradientLayer(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        sobel_x = np.array([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])
        sobel_y = sobel_x.T

        self.Gx = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding='same',bias=False)
        self.Gy = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding='same',bias=False)

        self.Gx.weight = nn.Parameter(torch.tensor(sobel_x,dtype=torch.float64).unsqueeze(0).unsqueeze(0))#.to(cfg['device']))
        self.Gy.weight = nn.Parameter(torch.tensor(sobel_y,dtype=torch.float64).unsqueeze(0).unsqueeze(0))#.to(cfg['device']))

        self.Gx.weight.requires_grad=False
        self.Gy.weight.requires_grad=False
        
        self.float()

    def forward(self,d):
        d=d.float()
        gx = self.Gx(d)
        gy = self.Gy(d)

        sq = torch.sqrt(torch.pow(gx,2) + torch.pow(gy,2))
        

        return sq
    
#loss = torch.mean(dist ** 2) - 0.5 / ((torch.numel(dist)) ** 2) * (torch.sum(dist) ** 2)

#image_gradient = image_gradient.eval()

class ScaleLoss(nn.Module):
    def __init__(self,image_gradient,**cfg):
        super(ScaleLoss,self).__init__()
        self.min_dist = cfg['min_dist']
        self.max_dist = cfg['max_dist']
        self.cfg = cfg
        self.image_gradient=image_gradient
        
    def forward(self,pred,target,**kwargs):

        mask = (target > self.cfg['min_dist']) & (target < self.cfg['max_dist'])

        
        dist = pred - target#torch.log(nn.functional.relu(pred)+.1) - torch.log(target+.1)

        mse = torch.mean(dist[mask] ** 2)
        scale = ((torch.sum(dist[mask]) ** 2) / (2*(torch.numel(dist[mask])) ** 2))
        loss = mse -  scale

        loss_grads = self.image_gradient(dist)

        loss_grads = loss_grads[mask]

        loss_grads = torch.nan_to_num(loss_grads, nan=0.0, posinf=0.0, neginf=0.0).mean()

        loss = loss + loss_grads

            
        print(f'[loss_components] mse: {mse}, scale: {scale}, grad: {loss_grads}')

        return {'loss':loss}
    

def training_epoch(epoch,train_dataloader,model,optimizer,device,loss_fn,args,log):
    
    accumulation_steps = 8
    
    model.train()

    num_iter = len(train_dataloader)
    
    running_loss = 0.0
    model.train_metrics.reset()
    for batch_idx,batch in enumerate(train_dataloader):
        
        
        inputs = batch['processed_image'].to(device)
        y_pred = model(inputs)['upsampled_logits']
        
        #import pdb;pdb.set_trace()
     
        loss = loss_fn(y_pred,batch['processed_mask'].to(device))['loss'] / accumulation_steps

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        
        running_loss+= (loss.item()*accumulation_steps)
        

        

        with torch.no_grad():

            model.train_metrics(y_pred,batch['processed_mask'].to(device))
            
            
            
             
        if (batch_idx+1)%accumulation_steps==0:
            
            optimizer.step()
            
            optimizer.zero_grad()
            
        train_metrics_d  = model.train_metrics.compute()      

        l1_metric = train_metrics_d["{}/train_MeanAbsoluteError5".format(args['experiment_version'])]
        
        ss = f'[epoch:{epoch} | iter:{batch_idx}/{num_iter}] train_loss: {running_loss / (1+batch_idx)} , l1: {l1_metric}'
        log.write(ss)
        print(ss)


                    
        
        
    train_metrics_dict = model.train_metrics.compute()
    

    return model , running_loss / (batch_idx+1), train_metrics_dict 


def validation_epoch(epoch,valid_dataloader,model,device,loss_fn,args,log):
    running_loss = 0.0
    model.valid_metrics.reset()
    
    num_iter = len(valid_dataloader)

    with torch.no_grad():
        model = model.eval()
        
        for batch_idx,batch in enumerate(valid_dataloader):

            y_pred = model(batch['processed_image'].to(device))['upsampled_logits']

            loss = loss_fn(y_pred,batch['processed_mask'].to(device))['loss']

            if batch_idx==0:

                log.saveimages(y_pred,batch['processed_mask'],batch['processed_image'])

            running_loss+=loss.item()
            
            model.valid_metrics(y_pred,batch['processed_mask'].to(device))
            
            valid_metrics_d = model.valid_metrics.compute()
            l1_metric = valid_metrics_d["{}/valid_MeanAbsoluteError5".format(args['experiment_version'])]
            ss = f'[epoch:{epoch} | iter:{batch_idx}/{num_iter}] valid_loss: {running_loss / (1+batch_idx)} , l1: {l1_metric}'        
            log.write(ss)
            print(ss)

    valid_metrics_dict = model.valid_metrics.compute()
    
          
    return model , running_loss / (batch_idx+1), valid_metrics_dict 

def test(test_dataloader,model,loss_fn):
    model = model.eval()
    num_iter = len(test_dataloader)
    with torch.no_grad():
        model = model.eval()
        model.test_metrics.reset()
        
        for batch_idx,batch in enumerate(valid_dataloader):

            loss = model.test_step(batch, batch_idx)['loss']
            
            running_loss+=loss
            
            print(f'[test:0 | iter:{batch_idx}/{num_iter}] valid_loss: {running_loss / (1+batch_idx)}')
    tes_metrics_dict = model.test_metrics.compute()
    
    return running_loss / (batch_idx+1), tes_metrics_dict

from xlightning.data.datamodules import get_inverse_preprocessing_fn

def write_dict_json(filename,d):
    filename = open(filename,'w')
    json.dump(d,filename)
    pass

class Trainer:
    def __init__(self,args,model_name,device):
        self.device=device
        self.args = args
        image_gradient = GradientLayer(args)
        image_gradient = image_gradient.to(device)
        self.loss_fn = ScaleLoss(image_gradient,**args)#L1Loss(**args)
        inv_fn = get_inverse_preprocessing_fn(**args)
        self.logger = Log(model_name ,'train',inv_fn)
        self.model_name = model_name
        self.root = os.environ['ROOT']

    def _training_epoch(self,epoch,train_dataloader,model,optimizer,device):

        return training_epoch(epoch,train_dataloader,model,optimizer,device,self.loss_fn,self.args,self.logger)
        
    def train(self,train_dataloader,valid_dataloader=None,model=None,device='cuda'):
        train_loss_curve=[]
        valid_loss_curve=[]
        best_model=None
        best_loss=1000.
        earlystopping_counter = 0
        PATH=f'{self.root}/{self.model_name}_best60_train_scale_adam.pt'
        
        optimizer = model.configure_optimizers()['optimizer']#torch.optim.Adam(model.parameters(),lr=model.hparams['lr'])
        
        epoch = 0
        while True:
        

            model , train_loss, train_metrics_dict = self._training_epoch(epoch,train_dataloader,model,optimizer,device)

            if valid_dataloader: 
                model , valid_loss, valid_metrics_dict = self._validation_epoch(epoch,valid_dataloader,model,device)
            
                l1_metric = valid_metrics_dict["{}/valid_MeanAbsoluteError5".format(self.args['experiment_version'])]
            else:
                l1_metric=False
            self.logger.update(train_loss,valid_loss = valid_loss if valid_dataloader else None,epoch = epoch,l1=l1_metric)

            self.logger.plot(l1_metric)
                    
            if valid_loss<best_loss:
                earlystopping_counter=0.0
                best_model_sd = model.state_dict()
                best_loss = valid_loss
                metric_dict = model.valid_metrics.compute()
                for k,v in metric_dict.items():
                    metric_dict[k]=v.item()
                metric_dict['epoch']=epoch
                
                torch.save({'logs':self.logger,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metric_dict,
                }, PATH)

                write_dict_json('{}.json'.format(PATH.split('.')[0]),metric_dict)
            
            if earlystopping_counter>15:
                break

            earlystopping_counter+=1
            epoch+=1
                
        return best_model_sd,train_metrics_dict,valid_metrics_dict, optimizer        


    def trainval(self,num_epochs,train_dataloader,valid_dataloader=None,model=None,device='cuda'):
        train_loss_curve=[]
        valid_loss_curve=[]
        best_model=None
        best_loss=1000.
        earlystopping_counter = 0
        PATH=f'{self.root}/{self.model_name}_best60_trainval_scale_adam_e{num_epochs}.pt'
        
        optimizer = model.configure_optimizers()['optimizer']#torch.optim.Adam(model.parameters(),lr=model.hparams['lr'])
        
        
        for epoch in range(num_epochs):
        

            model , train_loss, train_metrics_dict = self._training_epoch(epoch,train_dataloader,model,optimizer,device)

            if valid_dataloader: 
                model , valid_loss, valid_metrics_dict = self._validation_epoch(epoch,valid_dataloader,model,device)
            
                l1_metric = valid_metrics_dict["{}/valid_MeanAbsoluteError5".format(self.args['experiment_version'])]
            else:
                l1_metric=False
            self.logger.update(train_loss,valid_loss = valid_loss if valid_dataloader else None,epoch = epoch,l1=l1_metric)

            self.logger.plot(l1_metric)

                
            torch.save({'logs':self.logger,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': model.valid_metrics.compute(),
                }, PATH)
            

                
        return train_metrics_dict,valid_metrics_dict, optimizer        



    def _validation_epoch(self,epoch,valid_dataloader,model,device):
        return validation_epoch(epoch,valid_dataloader,model,device,self.loss_fn,self.args,self.logger)
        

    def load_model_dm(self,mode,device,args):
        return load_model_dm(mode,device,args)

import pandas as pd



def load_args(model_name,experiment_version,cuda):
    df = pd.read_csv('/petrobr/algo360/current/MultiGPU-lightning/results1.csv')

    #df = df[(df.loss=='l1')&((df.max_dist==60)|(df.max_dist==5))].fillna(value='None')

    df = df[(df.loss=='l1')&((df.max_dist==60))].fillna(value='None')

    summary = df[df.max_dist==60].groupby('model').min()[['experiment_version','SIZE','test_MeanAbsoluteError']].sort_values(by='test_MeanAbsoluteError')

    print(summary)
    args = df[df.experiment_version == experiment_version]

    for _,row in args.iterrows():
        break

    args = row.to_dict()

    for k,v in args.items():
        if v=='None':
            args[k]=None

    try:
        args['features']=int(args['features'])
    except:
        print('no features')



    try:
        args['features']=int(args['features'])
    except:
        print('no features')

    try:
        args['n_bins']=int(args['n_bins'])
    except Exception as e:
        print(e)

    try:
        args['bts_size']=512
    except Exception as e:
        print(e)


    args['batch_size']=15

    args['pretrained_kitti']=True

    args['path2meta']='/petrobr/algo360/current/dataset_1_cube/consolidated_meta.csv'

    args['optimizer']='adam'

    args['num_trainloader_workers']=0
    args['num_validloader_workers']=0


    
    return args
    
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import plotly.express as px
from PIL import Image
import os
class Log:
    def __init__(self,filename,mode,inv_fn):
        self.train_loss_curve=[]
        self.valid_loss_curve=[]
        self.l1_curve=[]
        self.epochs=[]
        self.epoch_counter=0
        self.f = f'{filename}_{mode}'
        self.mode = mode
        self.inv_fn = inv_fn
        self.root = os.environ['ROOT']
        
    
    def update(self,train_loss,valid_loss=None,epoch=None,l1=None):
        self.train_loss_curve.append(train_loss)

        if isinstance(valid_loss,float): 
            self.valid_loss_curve.append(valid_loss)
            self.l1_curve.append(l1)
        if isinstance(epoch,int):
            self.epochs.append(epoch)
        else:
            self.epochs.append(self.epoch_counter)
            self.epoch_counter+=1
            
    def write(self,s):
        with open(f'{self.f}.txt','w') as file:
            file.writelines(f'{s}\n')
            
    def saveimages(self,y_pred,y_target,x):
        with torch.no_grad():
            max_range = int(y_target.max().detach().numpy().item())
            mask = (y_target > 60) | (y_target < .02)
            y_pred[mask]=0.0#max_range
            y_target[mask]=0.0#max_range
            y_pred = torchvision.utils.make_grid(y_pred.cpu(), nrow=5, normalize=True, value_range=(0,max_range)).permute(1,2,0)
            y_target = torchvision.utils.make_grid(y_target.cpu(), nrow=5, normalize=True, value_range=(0,max_range)).permute(1,2,0)
            x = torchvision.utils.make_grid(self.inv_fn(x.cpu()), nrow=5).permute(1,2,0)


            cmap = plt.cm.get_cmap('plasma')

            Image.fromarray((255*cmap(y_pred[:,:,0])).astype(np.uint8)).save(f'{self.root}/log_images/prediction_{self.f}.png')
            
            Image.fromarray((255*cmap(torch.abs(y_target[:,:,0]-y_pred[:,:,0]))).astype(np.uint8)).save(f'{self.root}/log_images/error_{self.f}.png')

            Image.fromarray((255*cmap(y_target[:,:,0])).astype(np.uint8)).save(f'{self.root}/log_images/target_{self.f}.png')

            Image.fromarray((255*x.detach().numpy()).astype(np.uint8)).save(f'{self.root}/log_images/inputs_{self.f}.png')

        
    def plot(self,l1=True):
        plt.close()
        clear_output(wait=True)
        best_curve = self.l1_curve if l1 else self.valid_loss_curve
        if len(best_curve)!=0: 
            best_idx = np.argmin(best_curve)
            plt.title('loss curves, best l1: {:.2f}'.format(best_curve[best_idx]))
        else:
            plt.title('loss curves')
                    
        plt.plot(self.epochs,self.train_loss_curve,marker='o',label='train')
        if len(self.valid_loss_curve)!=0: plt.plot(self.epochs,self.valid_loss_curve,marker='o',label='valid')
        plt.legend()
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        
        mode = self.f
        plt.savefig(f'{self.root}/log_images/{mode}_curves.png')
        plt.show() 


            


    
def test_logging():
    log = Log(f'{model_name}.txt','train')
    for train_loss,valid_loss in zip(np.arange(10,),np.arange(10.,)*2.):     
        log.update(train_loss,valid_loss=valid_loss)
        log.plot()
    pass


def train(num_epochs,train_dataloader,valid_dataloader=None,model=None,device='cuda'):
    train_loss_curve=[]
    valid_loss_curve=[]
    best_model=None
    best_loss=1000.
    earlystopping_counter = 0
    
    optimizer = model.configure_optimizers()['optimizer']#torch.optim.Adam(model.parameters(),lr=model.hparams['lr'])
    
    epoch = 0
    while True:
    

        model , train_loss, train_metrics_dict = training_epoch(epoch,train_dataloader,model,optimizer,device)

        if valid_dataloader: model , valid_loss, valid_metrics_dict = validation_epoch(epoch,valid_dataloader,model,device)

        log.update(train_loss,valid_loss = valid_loss if valid_dataloader else None,epoch = epoch)

        log.plot()
        
        earlystopping_counter+=1
        
        if valid_loss<best_loss:
            earlystopping_counter=0.0
            best_model_sd = model.state_dict()
            best_loss = valid_loss
            
            torch.save({'logs':log,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': model.valid_metrics.compute(),
            }, PATH)
        
        if earlystopping_counter>10:
            break
        epoch+=1
            
    return best_model_sd,log,train_metrics_dict,valid_metrics_dict, optimizer
    

def load_model_dm(mode,device,args):
    dm = xld.dm_factory(**args)(**args)

    dm.setup(mode=mode)
    dm.setup_folds(1)
    dm.setup_fold_idx(0)

    print('[loading dataloaders]')

    train_dataloader = dm.train_dataloader()
    train_dataloader
    
    valid_dataloader=None
    if mode=='train':
        valid_dataloader = dm.val_dataloader()
    elif mode=='trainval':
        valid_dataloader = dm.test_dataloader()
    else:
        raise

    def get_model(**args):
        model = {'dpt':DPTmodel,
                    'dpt2':DPTmodel2,
                            'glp':GLPmodel,
                            'newcrf':NewCRF,
                            'adabins':AdaBins,
                            'bts':BTS,
                     'deeplabv3plusdepthfine':DeepLabV3plusDepthFine,
                     'deeplabv3plusdepth':DeepLabV3plus_Depthcustom,
                     'deeplabv3plus':DeepLabV3plus_custom,
                     'fcn':FCNModel,
                     'gan':GAN,
                     'ganfine':FineGAN
                     }[args['model']](**args)

        if args['gans_wrapper']:
            model = gans_wrapper.GAN(model,**args).float()
        return model

    print('[loading model]')
    model = get_model(**args)
    model.to(device)


    
    model.log = lambda *args,**kwargs:None
    model.log_dict = lambda *args,**kwargs:None
    return model,dm,train_dataloader,valid_dataloader



        
    

