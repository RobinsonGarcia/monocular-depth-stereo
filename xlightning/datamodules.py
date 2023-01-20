#%%
from __future__ import print_function, division
import os

from numpy.core.numeric import indices

import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler, BatchSampler
from torchvision import transforms, utils
import torchvision
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import StratifiedKFold

from xlightning.depthmap import depthmap_phi_lamb
from xlightning.depth_map_utils import fill_in_fast

def load_transformations(**cfg):


        H,W = cfg['SIZE'],cfg['SIZE']
        HA = cfg['heavy_aug']

        train_transform = A.Compose([\
               #Removed to corr: A.augmentations.transforms.Equalize(),
              #A.augmentations.geometric.resize.RandomScale(scale_limit=(-.1,.1),always_apply=True, interpolation=cv2.INTER_AREA),\
               #A.augmentations.geometric.resize.SmallestMaxSize(max_size=H,interpolation=cv2.INTER_AREA,always_apply=False, p=1.0),\
              #A.augmentations.crops.transforms.CenterCrop(H,W,always_apply=False, p=1.0),
              #A.augmentations.crops.transforms.RandomCrop (H,H, always_apply=False, p=1.0), # added to corr
               #A.augmentations.geometric.rotate.Rotate(limit=45,interpolation=cv2.INTER_AREA, border_mode=4, value=None, mask_value=None, method='largest_box', crop_border=False, always_apply=False, p=0.5),\
              A.augmentations.transforms.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),\
               A.augmentations.transforms.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=False, p=0.5),
                A.Cutout(always_apply=False, p=.8, num_holes=12, max_h_size=15, max_w_size=15),
               A.augmentations.dropout.coarse_dropout.CoarseDropout(max_holes=15, max_height=45, max_width=45, min_holes=12, min_height=8, min_width=8, fill_value=0, mask_fill_value=None, always_apply=False, p=0.9),
               #KFOLD3-RemoverforKFOLD4A.augmentations.dropout.coarse_dropout.CoarseDropout(max_holes=100, max_height=45, max_width=45, min_holes=60, min_height=20, min_width=8, fill_value=0, mask_fill_value=None, always_apply=True),
                A.ISONoise(always_apply=False, p=.6, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
                    A.RandomBrightnessContrast(p=.6),    
                    A.RandomGamma(p=.7),  
                    #KFOLD3-RemoverforKFOLD4A.MotionBlur(always_apply=False, p=.3, blur_limit=(3, 7)),
                    A.HorizontalFlip(.5), #==> added for corr
                    #A.augmentations.geometric.rotate.Rotate(limit=45, interpolation=cv2.INTER_AREA, always_apply=False, p=0.8) #==> added for corr
              ])

        normalization = {
            'mean':(0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5)
        }
        max_pixel_value=255.0

        preprocessing_fn = A.Compose([
            A.Normalize(
            mean=normalization['mean'],
            std=normalization['std'],
            max_pixel_value=max_pixel_value)]
            )

        test_transform  = A.Compose(
            [ A.augmentations.geometric.resize.SmallestMaxSize(max_size=cfg['resize_H_to'],interpolation=cv2.INTER_AREA,always_apply=False, p=1.0),\
              A.augmentations.crops.transforms.CenterCrop(cfg['resize_H_to'],cfg['resize_H_to'],always_apply=False, p=1.0),
            ]
            )
        return preprocessing_fn , train_transform, test_transform

def get_inverse_preprocessing_fn(**cfg):



    default_cfg ={
                'input_size': (3, 224, 224),
                'crop_pct': 0.875,
                'interpolation': 'bicubic',
                #'mean': (0.485, 0.456, 0.406),
                #'std': (0.229, 0.224, 0.225),
                'std': (0.5, 0.5, 0.5),
                'mean':(0.5, 0.5, 0.5),
                }

    mean = -np.array(default_cfg['mean'])/np.array(default_cfg['std'])
    std = 1/np.array(default_cfg['std'])
    inverse_preprocessing_fn = torchvision.transforms.Normalize(mean=mean,std=std)
    return inverse_preprocessing_fn

def load_files(mode,cfg):
    kfold = 'kfold{}'.format(cfg['fold'])
    df = pd.read_csv(cfg['path2meta'])

    if mode=="trainvaltest":
            
        files = df.saveas.tolist()
        print('trainvaltest: {}'.format(len(files)))
        return files

    if mode=='train':
        bool_mask = df[kfold]==1
    else:
        bool_mask = df[kfold]==0

    files = df[bool_mask].saveas.tolist()

    if not cfg['extended']:
        remain=[]
        for f in files:
            if 'cube' in f:
                remain.append(f)

        print('removed fibb50 {} files'.format(len(files)-len(remain)))
        files = remain
   
    print(f'[{mode},{kfold}]: {len(files)} files')    
    return files

class DepthDataset(Dataset):

    def load_filenames(self,mode):
        

        files = load_files(mode,self.cfg)
        return [os.path.join(self.cfg['npz_dir'],f.split('/')[-1]) for f in files]
        
    def __init__(
            self, 
            mode,
            transform=None,
            preprocessing_fn=None,
            **cfg):
        super().__init__()

      
        self.cfg = cfg
        self.npz_fps = self.load_filenames(mode) 
        np.random.seed(1234)
        np.random.shuffle(self.npz_fps)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        
    def __len__(self):
        return len(self.npz_fps)

    def adjust_depthmap(self,R):
        a_phi, a_theta  =  depthmap_phi_lamb(R)
        yy =  R * np.cos(a_phi)*np.cos(a_theta)
        return yy

    def __getitem__(self, idx):

        arr = np.load(self.npz_fps[idx])

        rgb_sample =  arr['image']
        depth_sample = arr['depthmap']

        if self.cfg['adjust_depth']: 
            depth_sample = self.adjust_depthmap(depth_sample)

        if self.cfg['fill_in']:
            depth_sample = fill_in_fast(np.float32(depth_sample),max_depth=self.cfg['max_dist'])
            depth_sample[depth_sample==0]=self.cfg['max_dist']-1

        
        HH = self.cfg['resize_H_to']
        H,W,_=rgb_sample.shape
        factor = W/H
        WW = int(HH*factor)

        rgb_sample = cv2.GaussianBlur(rgb_sample,(5,5),0)
        rgb_sample = cv2.resize(rgb_sample, (HH,WW),interpolation= cv2.INTER_AREA)
        depth_sample = cv2.resize(depth_sample, (HH,WW),interpolation= cv2.INTER_NEAREST) 
        

        if self.transform:
            if not self.cfg['no_transform']:
                sample = self.transform(image=rgb_sample,mask=depth_sample)
                rgb_sample, depth_sample = sample['image'], sample['mask']
 
        
        if self.preprocessing_fn:
            sample = self.preprocessing_fn(image=rgb_sample,depth=depth_sample)
            rgb_sample = sample['image']

        rgb_sample = np.ascontiguousarray(rgb_sample).astype(np.float32)
        depth_sample = np.ascontiguousarray(depth_sample.astype(np.float32))

        rgb_sample = rgb_sample.transpose(2,0,1)
        depth_sample = depth_sample[np.newaxis,:,:]

        if self.cfg['mask_background']:
            mask = (depth_sample < self.cfg['min_dist']) | (depth_sample > self.cfg['max_dist'])
            rgb_sample[0,mask[0,:,:]]=0.0000
            rgb_sample[1,mask[0,:,:]]=0.0000
            rgb_sample[2,mask[0,:,:]]=0.0000
                
        return {'processed_image':rgb_sample,'processed_mask':depth_sample,'filename':(self.npz_fps[idx],'none')}
        

class DepthDataModule(pl.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("dm")

        
        parser.add_argument("--batch_size", type=int, default=15)
        parser.add_argument("--SIZE", type=int, default=384)
        parser.add_argument("--validation_module", type=str, default='MD-08_missing_files')
        parser.add_argument("--path2meta", type=str, default='/nethome/algo360/mestrado/dataset_final/consolidated_meta_final_fibb_cube_folds.csv')
        parser.add_argument("--npz_dir", type=str, default='/nethome/algo360/mestrado/dataset_final/npz_compressed_384')
        parser.add_argument("--fold", type=int, default=1)
        parser.add_argument("--num_trainloader_workers", type=int, default=6)
        parser.add_argument("--num_validloader_workers", type=int, default=6)
        parser.add_argument("--resize_H_to", type=int, default=384)

        parser.add_argument("--extended", action='store_true')
        parser.add_argument("--heavy_aug",action='store_true')
        parser.add_argument("--random_split", action='store_true')
        parser.add_argument("--adjust_depth", action='store_true')
        parser.add_argument("--mask_background",action='store_true')
        parser.add_argument("--fill_in", action='store_true')
        parser.add_argument("--no_transform", action='store_true')

        #parser.add_argument("--extended", type=(lambda x:(x).lower()=='true'), default=False)
        #parser.add_argument("--heavy_aug", type=(lambda x:(x).lower()=='true'), default=False)
        #parser.add_argument("--random_split", type=(lambda x:(x).lower()=='true'), default=False)
        #parser.add_argument("--adjust_depth", type=(lambda x:(x).lower()=='true'), default=False)
        #parser.add_argument("--mask_background", type=(lambda x:(x).lower()=='true'), default=False)
        #parser.add_argument("--fill_in", type=(lambda x:(x).lower()=='false'), default=True)
        
        return parent_parser
      
    def __init__(self,**cfg):
        super().__init__()

        self.cfg = cfg
        self.batch_size = cfg['batch_size']
        
        self.preprocessing_fn , self.train_transform, self.test_transform = load_transformations(**cfg)

    def setup(self,stage=None,mode=None):
        self.corr_train = DepthDataset(mode='train' if not mode else mode,transform=self.train_transform, preprocessing_fn=self.preprocessing_fn,**self.cfg)
        self.corr_val = DepthDataset(mode='val',transform=self.test_transform, preprocessing_fn=self.preprocessing_fn,**self.cfg)
        self.corr_test = DepthDataset(mode='test',transform=self.test_transform, preprocessing_fn=self.preprocessing_fn,**self.cfg)

    def train_dataloader(self):
        #try:
        #    sampler = DistributedSampler(self.corr_train,num_replicas=self.num_replicas,shuffle=True,) if 'dp' in self.cfg['strategy'] else None    
        #except:
        #    sampler=None
        return DataLoader(self.corr_train,batch_size=self.batch_size,drop_last=True , num_workers=self.cfg['num_trainloader_workers'],pin_memory=True)

    def val_dataloader(self):
        #try:
        #    sampler = DistributedSampler(self.corr_val,num_replicas=self.num_replicas) if 'dp' in self.cfg['strategy'] else None   
        #except:
        #    sampler=None
        return DataLoader(self.corr_val,  batch_size=self.batch_size,drop_last=True , num_workers=self.cfg['num_validloader_workers'],pin_memory=True)
        
    def test_dataloader(self):
        #try:
        #    sampler = DistributedSampler(self.corr_test,num_replicas=self.num_replicas) if 'dp' in self.cfg['strategy'] else None   
        #except:
        #    sampler=None
        return DataLoader(self.corr_test,  batch_size=self.batch_size,drop_last=True , num_workers=self.cfg['num_validloader_workers'],pin_memory=True)
    

    