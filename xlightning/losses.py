import torch
import torch.nn as nn
import numpy as np
from xlightning.depthmap import depthmap_to_3D
import torchmetrics
class BaseLoss(nn.Module):
    def __init__(self,**cfg):
        super(BaseLoss,self).__init__()
        self.min_dist = cfg['min_dist']
        self.max_dist = cfg['max_dist']
        self.cfg = cfg


class ScaleInvariantLoss(nn.Module):
    def __init__(self,**cfg):
        super().__init__()
        print('initializing scale invariant loss')

        self.min_dist = cfg['min_dist']
        self.max_dist = cfg['max_dist']
        self.cfg = cfg

        sobel_x = np.array([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])
        sobel_y = sobel_x.T
        self.sobel_x = torch.tensor(sobel_x[None,None,:,:])
        self.sobel_x.requires_grad=False
        self.sobel_y = torch.tensor(sobel_y[None,None,:,:])
        self.sobel_y.requires_grad=False

        #self.Gx = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding='same',bias=False)
        #self.Gy = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding='same',bias=False)

        #self.Gx.weight = nn.Parameter(torch.tensor(sobel_x,dtype=torch.float64).unsqueeze(0).unsqueeze(0))#.to(cfg['device']))
        #self.Gy.weight = nn.Parameter(torch.tensor(sobel_y,dtype=torch.float64).unsqueeze(0).unsqueeze(0))#.to(cfg['device']))

        #self.Gx.weight.requires_grad=False
        #self.Gy.weight.requires_grad=False
        


    def forward(self,pred,target,**kwargs):

        a_max = self.cfg['max_dist']
        a_min = self.cfg['min_dist']

        mask = (target < a_min) | (target > a_max) 

        if not self.cfg['no_disparity']:
            mask = (target > 1./a_min) | (target < 1./a_max) 
            target[~mask] = 1/target[~mask]
            #target[mask] = 1/(self.cfg['max_dist']-1)
            #target = (target - self.cfg['disp_shift']) / self.cfg['disp_scale']

        #target[mask]=0.
        #mask = target < a_min
        #target[mask] = 0.01
        #mask = target > 1000


        #==> DEBUG 

        if not self.cfg['no_pred_log']:
            dist = pred[~mask] - torch.log(target[~mask])
        else: 
            dist = torch.log(pred[~mask]+1e-10) - torch.log(target[~mask])


        
        
        """
        if self.cfg['disparity']:
            dist = torch.log(torch.clamp(pred,min=.01)) - torch.log(torch.clamp(target,min=.01))#pred - target#
            print('log loss dist',torch.max(dist.reshape(-1)))
            #dist = pred - target#pred - target#
        else:
            dist = pred - target#pred - target#
        """

        #dist[torch.max(dist**2)>0.8*torch.max(dist**2)]=0

        
        #==> DEBUG print('loss',torch.max(torch.abs(dist)),torch.max(target),torch.max(pred))
        #with torch.no_grad():
        #print('l1',torch.mean(torch.abs(dist[~mask])))
        mse = torch.mean(dist ** 2)
        scale = ((torch.sum(dist) ** 2) / ((torch.numel(dist)) ** 2))
        loss = mse -  torch.tensor(.85)*scale

        #if self.cfg['model']=='newcrf':
        loss = torch.tensor(10)*torch.sqrt(loss+torch.tensor(1e-5))


        #print('loss',loss)

        """
        if self.cfg['extend_3d']:
            
            pred_xyz = depthmap_to_3D(pred)
            target_xyz = depthmap_to_3D(target)

            

            pred_xyz = pred_xyz.permute(0,2,3,1)[mask[:,0],:]
            target_xyz =target_xyz.permute(0,2,3,1)[mask[:,0],:]


            mask2 = target_xyz !=0
            dist2 = pred_xyz[mask2] - torch.log(target_xyz[mask2])

            mse2 = torch.mean(dist2 ** 2)
            scale2 = ((torch.sum(dist2) ** 2) / ((torch.numel(dist2)) ** 2))
            loss2 = mse2 -  torch.tensor(.5)*scale2

            loss = torch.tensor(0.5)*loss + torch.tensor(0.5)*loss2
            
        
        d=dist

        #self.Gx.weight.type_as(pred)
        #self.Gy.weight.type_as(pred)

        #gx = self.Gx(d)
        #gy = self.Gy(d)
        wx = self.sobel_x.type_as(pred)
        wy = self.sobel_y.type_as(pred)
        wx.requires_grad=False
        wy.requires_grad=False
        gx = torch.nn.functional.conv2d(d, wx,padding='same')
        gy = torch.nn.functional.conv2d(d, wy,padding='same')

        loss_grads = torch.sqrt(torch.pow(gx,2) + torch.pow(gy,2))

        loss_grads = loss_grads[~mask]


        
        loss_grads = torch.nan_to_num(loss_grads, nan=0.0, posinf=0.0, neginf=0.0).mean()
        print('loss_grad',loss_grads)
        loss = loss + loss_grads
        """
        # print('loss',loss)

        return {'total_loss':loss+0.0,
        'loss':loss,
        'texture_loss':0.0}

# https://arxiv.org/pdf/1606.00373.pdf
class berHu(BaseLoss):
  
    def forward(self, pred, target,**kwargs):
        
        # mask out zero values and invalid regions
       
        mask = (target<self.cfg['min_dist'])|(target>self.cfg['max_dist'])
        target = target[~mask]
        pred = pred[~mask]

        if self.cfg['disparity']:
            target = 1./target


        if self.cfg['add_texture_head']:
            texture_loss = torch.tensor(1000)*torch.nn.functional.mse_loss(kwargs['texture_logits'],kwargs['texture_labels'])
        else:
            texture_loss=0.0

        if self.cfg['extend_3d']:
            pred = depthmap_to_3D(pred,self.cfg['adjust_depth'])
            target = depthmap_to_3D(target,self.cfg['adjust_depth'])

        loss= torch.nn.functional.smooth_l1_loss(pred,target,reduction='none')
     
    
        loss = torch.mean(loss)

        total_loss = loss + texture_loss
        
        return {'total_loss':total_loss,
        'loss':loss,
        'texture_loss':texture_loss}


class L1Loss(BaseLoss):

    def forward(self, pred,target,**kwargs):
        
        if self.cfg['extend_3d']:
            if self.cfg['log_scale']:
                raise
            pred = depthmap_to_3D(pred,self.cfg['adjust_depth'])
            target = depthmap_to_3D(target,self.cfg['adjust_depth'])

 
        mask = (target<self.cfg['min_dist'])|(target>self.cfg['max_dist'])
        target[mask] = 0.
        mask = target > 1000
        target = target[~mask]
        pred = pred[~mask]


        if self.cfg['log_scale']:
            target = torch.log(target)

        if self.cfg['disparity']:
            target = 1./target


        loss = torch.nn.functional.l1_loss(pred,target)#,reduction='none') 
        
        total_loss = loss 

        return {'total_loss':total_loss,
        'loss':loss,
        'texture_loss':0.0}

class MSELoss(BaseLoss):

    def forward(self, pred,target,**kwargs):


        if self.cfg['extend_3d']:
            if self.cfg['log_scale']:
                raise
            pred = depthmap_to_3D(pred,self.cfg['adjust_depth'])
            target = depthmap_to_3D(target,self.cfg['adjust_depth'])


        mask = (target<self.cfg['min_dist'])|(target>self.cfg['max_dist'])
        target = target[~mask]
        pred = pred[~mask]

        if self.cfg['log_scale']:
            target = torch.log(target)

        if self.cfg['disparity']:
            target = 1./target


        if self.cfg['add_texture_head']:
            raise
            texture_loss = torch.tensor(1000)*torch.nn.functional.mse_loss(kwargs['texture_logits'],kwargs['texture_labels'])
        else:
            texture_loss=0.0


        loss = torch.nn.functional.mse_loss(pred,target)#,reduction='none')


        total_loss = loss + texture_loss

        return {'total_loss':total_loss,
        'loss':loss,
        'texture_loss':texture_loss}


def load_MonocularDepthLosses(**cfg):
  
        available_losses = {'l1':L1Loss,
        'l2':MSELoss,
        'scale_invariant':ScaleInvariantLoss,
        'scale_invariant_gradient':ScaleInvariantLossGradient,
        'berHu':berHu}
        return available_losses[cfg['LOSS']](**cfg)



# based on:
#==> https://hal.archives-ouvertes.fr/hal-01925321/document
#==> https://arxiv.org/pdf/1907.01341.pdf
#==> https://arxiv.org/pdf/2103.13413.pdf
#==> https://github.com/zhezh/focalloss/blob/master/focalloss.py

