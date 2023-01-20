import sys
sys.path.insert(1,'/nethome/algo360/mestrado/monocular-depth-estimation/')

import numpy as np
import torch
import argparse
from xlightning.datamodules import DepthDataModule
from xlightning.models.pl_dpt import DPTmodel
import os

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str)

parser = DepthDataModule.add_model_specific_args(parser)

parser = DPTmodel.add_model_specific_args(parser)

cfg = vars(parser.parse_args())

cfg['pretrained_depth']=False
cfg['max_dist'] = 100.
cfg['min_dist'] = .02
cfg['experiment_version'] = '1234'
cfg['model'] = 'dpt_hybrid'
cfg['pretrained_dataset'] = 'midas'
cfg['batch_size'] = 10
cfg['extended']=True
cfg['fill_in']=True
cfg['no_transform']=True

saveto = '/nethome/algo360/mestrado/monocular-depth-estimation/xlightning/models/dpt/weights/dpt_{}_fold_{}.npz'.format(cfg['pretrained_dataset'],cfg['fold'])


dm = DepthDataModule(**cfg)
dm.setup()
train_dataloader = dm.train_dataloader()
valid_dataloader = dm.val_dataloader()
device = cfg['device']

class MeanScaleShift:
    def __init__(self,device):
        self.reset()

    def update(self,pred_disparity,target_disparity):

        A = torch.vstack([target_disparity,torch.ones_like(target_disparity)]).T

        scale_shift = torch.linalg.lstsq(A,pred_disparity).solution

        self.sum = .9*self.sum + .1*scale_shift
        self.total +=1
    def compute(self):
        return self.sum / self.total
    def reset(self):
        self.sum = torch.tensor([0.,0.])
        self.total = 0.

class RunningMSE:
    def __init__(self):
        self.reset()

    def update(self,pred_disparity,target_disparity,scale_shift):

        A = torch.vstack([target_disparity,torch.ones_like(target_disparity)]).T
        mse = torch.mean( (torch.sum(A*scale_shift,axis=1) - target_disparity)**2)
        self.mse_sum +=mse
        self.total+=1

    def compute(self):
        return self.mse_sum / self.total
    def reset(self):
        self.mse_sum = 0.0
        self.total = 0


mean_scaleshift = MeanScaleShift(device)
running_mse = RunningMSE()
val_running_mse = RunningMSE()

model = DPTmodel(**cfg)

model.eval()

for p in model.parameters():
    p.requires_grad=False

model.to(device)
print(cfg)

best_mse = 1e10
best_scale_shift = None
epoch = 1
count = 0
warm_up = 30
while True:
    for t,sample in enumerate(train_dataloader):

        img = sample['processed_image'].to(device)

        target_depthmap = sample['processed_mask'].to(device)

        pred_disparity= model(img)

        mask = (target_depthmap > cfg['max_dist']) | (target_depthmap < cfg['min_dist'])

        target_disparity = 1 / target_depthmap[~mask]

        pred_disparity = pred_disparity['upsampled_logits'][~mask]

        pred_disparity = pred_disparity.cpu()

        target_disparity = target_disparity.cpu()

        mean_scaleshift.update( pred_disparity, target_disparity)

        scale_shift = mean_scaleshift.compute()

        running_mse.update(pred_disparity , target_disparity, scale_shift)

        mse = running_mse.compute()

        scale_shift = scale_shift.numpy()

        if t%2==0:
            print('[TRAIN][MSE:{},epoch:{},iter:{}/{},device:{},fold:{}] Scale={}, Shift={} | valid_mse:{} | dist_max:{},dist_min:{})'.format(mse,epoch,t,len(train_dataloader),device,cfg['fold'],scale_shift[0],scale_shift[1],best_mse,target_depthmap[~mask].max(),target_depthmap[~mask].min()))
    
    for t,sample in enumerate(valid_dataloader):

        img = sample['processed_image'].to(device)

        target_depthmap = sample['processed_mask'].to(device)

        pred_disparity= model(img)

        mask = (target_depthmap > cfg['max_dist']) | (target_depthmap < cfg['min_dist'])

        target_disparity = 1 / target_depthmap[~mask]

        pred_disparity = pred_disparity['upsampled_logits'][~mask]

        pred_disparity = pred_disparity.cpu()

        target_disparity = target_disparity.cpu()

        val_running_mse.update(pred_disparity , target_disparity, scale_shift)

        mse = val_running_mse.compute()



        if t%2==0:print('[VALID][MSE:{},epoch:{},iter:{}/{},device:{},fold:{}] Scale={}, Shift={} | dist_max:{},dist_min:{})'.format(mse,epoch,t,len(valid_dataloader),device,cfg['fold'],scale_shift[0],scale_shift[1],target_depthmap[~mask],target_depthmap[~mask].max(),target_depthmap[~mask].min()))

    mse = val_running_mse.compute()

    if mse < best_mse:
        best_mse = mse
        best_scale_shift = scale_shift
        np.savez_compressed(saveto,scale=best_scale_shift[0],shift=best_scale_shift[1])
        count=0
    else:
        count+=1

    if count>5:
        print('*****************CONVERGED')
        break


    running_mse.reset()
    val_running_mse.reset()
    epoch+=1