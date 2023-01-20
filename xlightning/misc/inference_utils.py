import torch
import torchvision.transforms as transforms
#from xlightning.data.datamodules import algo360Dataset
from xlightning.models.segmentation.deeplabv3plus import DeepLabV3plus_custom
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class MultiScaleInference:
    def __init__(self,cfg={'root':'datasets/imagens_360/','input_file':'P76/input.json','resize_factor':3}):
        self.model = DeepLabV3plus_custom.load_from_checkpoint('best_model.ckpt')
        self.model.eval()
        self.dm = algo360Dataset(transform=True,**cfg)
        
    
        
    def step(self,x,idx,face_number):
        
        self.mu,self.out,self.rgb = self.compute(x,face_number,return_rgb=True)
        self.save_multiscale(self.mu,self.out,self.rgb,face_number,root='.tmp')

    def run_all(self,idx):
        x = self.dm[idx]
        self.step(x,idx,0)
        self.step(x,idx,1)
        self.step(x,idx,2)
        self.step(x,idx,3)
        self.step(x,idx,4)
        self.step(x,idx,5)
        
        

    def compute(self,x,face_number=0,return_rgb=True):
        H,W = x['finer_shape']
        images = x['images'][face_number]
        out = [torch.nn.functional.interpolate(torch.sigmoid(self.model(i)['upsampled_logits']),(H,W)) for i in images.values()]
        mu = torch.stack(out,dim=2)[0,0]
        mu = torch.mean(mu,dim=0)  
        if return_rgb:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            inv_norm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
            rgb_images = inv_norm(images[1][0]).permute(1,2,0)
            return mu,out, rgb_images
        return mu,out


    def save_multiscale(self,mu,out,rgb,face_number,root='.tmp'):
        cmap = plt.get_cmap('jet')
        [Image.fromarray((255*cmap(y[0,0].detach().numpy())[:,:,:3]).astype(np.uint8)).save('{}/tmp_face{}.png'.format(root,face_number)) for scale,y in enumerate(out)]
        Image.fromarray((255*cmap(mu.detach().numpy())[:,:,:3]).astype(np.uint8)).save('{}/tmp_face{}_mu.png'.format(root,face_number))
        Image.fromarray((255*rgb.detach().numpy()).astype(np.uint8)).save('{}/tmp_{}_rgb.png'.format(root,face_number))
        pass