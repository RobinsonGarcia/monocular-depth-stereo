
from xlightning.models.depth.base import BaseDepthModel
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from xlightning.models.custom_layers import *
from xlightning.models.custom_layers import _AtrousSpatialPyramidPoolingModule, DoNothing
from xlightning.losses import *
import torchvision
from collections import OrderedDict
from torch.optim import Adam,AdamW,SGD


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
        
        #features = [i+j for i,j in zip(y_rgb,y_depth)]
        features = y_rgb+y_depth

        return features
        
class DeepLabV3plusDepthFine(BaseDepthModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
 
        parser.add_argument("--encoder", type=str, default="resnet50")
        parser.add_argument("--pretrained", type=(lambda x:(x).lower()=='true'), default=True)
        parser.add_argument("--output_stride", type=int, default=16)
        parser.add_argument("--add_laws", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--add_classification_head", type=bool, default=False)
        parser.add_argument("--dropout_p", type=float, default=.5)
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--loss", type=str, default="scale")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--l2_reg", type=float, default=5e-4)
        parser.add_argument("--coarsenet_checkpoint", type=str, default='none')
        parser.add_argument("--min_dist", type=float, default=.2)
        parser.add_argument("--max_dist", type=float, default=150)
        parser.add_argument("--scale_invariant_ratio", type=float, default=.5)
        parser.add_argument("--disparity", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--log_scale", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--loss_cut", type=float, default=.8)
        return parent_parser
    
    def __init__(self,pretrained_encoders=True,**cfg):
        BaseDepthModel.__init__(self,pretrained_encoders=True,**cfg)
        
        if pretrained_encoders==False:
            self.hparams['coarsenet_checkpoint']=False
            self.hparams['pretrained']=False

        self.coarse_net = DeepLabV3plus_Depthcustom.load_from_checkpoint(self.hparams['coarsenet_checkpoint']) if isinstance(self.hparams['coarsenet_checkpoint'],str) else  DeepLabV3plus_Depthcustom(**self.hparams)
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
    @staticmethod
    def add_model_specific_args(parent_parser):
  
        parser = parent_parser.add_argument_group("model")
 
        parser.add_argument("--encoder", type=str, default="resnet200d")
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument("--output_stride", type=int, default=16)
        parser.add_argument("--add_texture_head", type=bool, default=False)
        parser.add_argument("--add_classification_head", type=bool, default=False)
        parser.add_argument("--dropout_p", type=float, default=.9)
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--loss", type=str, default="l1")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--l2_reg", type=float, default=5e-4)
        parser.add_argument("--min_dist", type=float, default=.2)
        parser.add_argument("--max_dist", type=float, default=150)
        parser.add_argument("--scale_invariant_ratio", type=float, default=.5)
        parser.add_argument("--disparity", type=bool, default=False)
        parser.add_argument("--log_scale", type=(lambda x:(x).lower()=='true'), default=False)
        parser.add_argument("--loss_cut", type=float, default=.8)
        parser.add_argument("--add_extra_decoder_layer", type=bool, default=False)
        parser.add_argument("--resize_logits", type=bool, default=True)
        parser.add_argument("--nodropout", type=bool, default=False)
        parser.add_argument("--nobatchnorm", type=bool, default=False)

        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--extend_3d", type=(lambda x:(x).lower()=='true'), default=False)

        parser.add_argument("--freeze_encoder", type=(lambda x:(x).lower()=='true'), default=False) #==> NOT IMPLEMENTED ON THIS MODEL
        parser.add_argument("--add_dropout", type=(lambda x:(x).lower()=='true'), default=False)  #==> NOT IMPLEMENTED ON THIS MODEL       

        parser.add_argument("--pretrained_kitti", type=(lambda x:(x).lower()=='true'), default=False)    

        return parent_parser
    
    def __init__(self,in_chans = 3,**cfg):
        BaseDepthModel.__init__(self,**cfg)

        self.encoder = timm.create_model(self.hparams['encoder'],
                                         features_only=True,
                                         pretrained=self.hparams['pretrained'],
                                         output_stride=self.hparams['output_stride'],
                                         in_chans=in_chans)   
        
        encoder_channels = self.encoder.feature_info.channels()
        self.aspp = _AtrousSpatialPyramidPoolingModule(in_dim=encoder_channels[-1], reduction_dim=256, output_stride=self.hparams['output_stride'], rates=[6, 12, 18] if self.hparams['output_stride']==16 else [12,24,36])
        self.bot_aspp = nn.Sequential(collections.OrderedDict([('conv1',nn.Conv2d(1280, 256, kernel_size=1, bias=False))]))
        self.bot_fine = nn.Sequential(collections.OrderedDict([('conv1',nn.Conv2d(encoder_channels[1 if self.hparams['output_stride']==8 else 2], 48, kernel_size=1, bias=False))]))
        
        self.interpolate = torch.nn.functional.interpolate
        self.dropout2D_1 = nn.Identity() if self.hparams['nodropout'] else nn.Dropout2d(self.hparams['dropout_p'])
        #self.dropout2D_2 = nn.Dropout2d(self.hparams['dropout_p']) if isinstance(self.hparams['dropout_p'],float) else DoNothing()

        if self.hparams['add_extra_decoder_layer']:
            self.bot_fine2 = nn.Sequential(collections.OrderedDict([('conv1_extra',nn.Conv2d(encoder_channels[0 if self.hparams['output_stride']==8 else 1], 24, kernel_size=1, bias=False))]))
   
            self.extra_decoder_layer = nn.Sequential(
            collections.OrderedDict([
            
            ('conv1_extra',nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False)),
            ('bn1_extra',nn.Identity() if self.hparams['nobatchnorm'] else nn.BatchNorm2d(256)),
            ('act1_extra',nn.ReLU(inplace=True)),
            ('dout1_extra',nn.Identity() if self.hparams['nodropout'] else nn.Dropout2d(self.hparams['dropout_p'])),
            ('conv2_extra',nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ('bn2_extra',nn.Identity() if self.hparams['nobatchnorm'] else nn.BatchNorm2d(256)),
            ('act2_extra',nn.ReLU(inplace=True)),
            ('dout1_extra',nn.Identity() if self.hparams['nodropout'] else nn.Dropout2d(self.hparams['dropout_p']))]))

        self.final_seg = nn.Sequential(
            collections.OrderedDict([
            
            ('conv1',nn.Conv2d(256+24 if self.hparams['add_extra_decoder_layer'] else 256 + 48, 256, kernel_size=3, padding=1, bias=False)),
            ('bn1',nn.Identity() if self.hparams['nobatchnorm'] else nn.BatchNorm2d(256)),
            ('act1',nn.ReLU(inplace=True)),
            ('dout1', nn.Identity() if self.hparams['nodropout'] else nn.Dropout2d(self.hparams['dropout_p'])),
            ('conv2',nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ('bn2',nn.Identity() if self.hparams['nobatchnorm'] else nn.BatchNorm2d(256)),
            ('act2',nn.ReLU(inplace=True)),
            ('dout2',nn.Identity() if self.hparams['nodropout'] else nn.Dropout2d(self.hparams['dropout_p'])),
            ('clf',nn.Conv2d(256, 1, kernel_size=1, bias= True))]))

        #nn.init.constant_(self.final_seg.clf.bias.data, 1.7)


        self.lr = self.hparams['lr']
        if self.hparams['add_texture_head']:
            self.texture_head = nn.Sequential(
            collections.OrderedDict([
            
            ('conv1',nn.Conv2d(256+24 if self.hparams['add_extra_decoder_layer'] else 256 + 48, 256, kernel_size=3, padding=1, bias=False)),
            ('bn1',nn.Identity() if self.hparams['nobatchnorm'] else nn.BatchNorm2d(256)),
            ('act1',nn.ReLU(inplace=True)),
            ('dout1', nn.Identity() if self.hparams['nodropout'] else nn.Dropout2d(self.hparams['dropout_p'])),
            ('conv2',nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ('bn2',nn.Identity() if self.hparams['nobatchnorm'] else nn.BatchNorm2d(256)),
            ('act2',nn.ReLU(inplace=True)),
            ('dout2',nn.Identity() if self.hparams['nodropout'] else nn.Dropout2d(self.hparams['dropout_p'])),
            ('clf',nn.Conv2d(256, 9, kernel_size=1, bias= True))]))


        if self.hparams['add_classification_head']:
            self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(encoder_channels [-1],self.hparams['num_classes'],1)
        )

        self.loss_fn =  {'scale':ScaleInvariantLoss, 'berHu': berHu, 'l1':L1Loss,'scale_gradient':ScaleInvariantLossGradient}[self.hparams['loss']](**cfg)#
   
    def forward(self,x):
        
        x_size = x.size() 
        

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

        if self.hparams['add_extra_decoder_layer']:
            dec0 = self.extra_decoder_layer(dec0)

            fmap_fine2 = fmaps[0 if self.hparams['output_stride']==8 else 1]

            fmap_coarse2_up = self.interpolate(dec0, fmap_fine2.size()[2:], mode='bilinear',align_corners=True)

            fmap_fine2 = self.bot_fine2(fmap_fine2)

            dec0 = torch.cat([fmap_coarse2_up,fmap_fine2], 1)
         
        logits = self.final_seg(dec0)

        if self.hparams['add_texture_head']:
            texture_logits = self.texture_head(dec0)
            upsampled_texture_logits = self.interpolate(texture_logits, x_size[2:], mode='bilinear',align_corners=True)
        else:
            texture_logits = None
            upsampled_texture_logits = None
        

        if self.hparams['log_scale']:
            logits = torch.nn.functional.relu(logits, inplace=True)
        
        upsampled_logits = self.interpolate(logits, x_size[2:], mode='bilinear',align_corners=True)

        if (self.hparams['log_scale'])|('scale' in self.hparams['loss']):
            return {'upsampled_logits':upsampled_logits,'clf_logits': None}
        else:    
            upsampled_logits = torch.nn.functional.relu(upsampled_logits) + .001

        return {'upsampled_logits':upsampled_logits,'clf_logits': None}


    def configure_optimizers(self):
        parameters = [
                #{'params': self.dpt.pretrained.parameters(),'lr':self.hparams['lr']*.01},
                {'params': self.parameters(), 'lr': self.hparams['lr']},
                
            ]
     
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

class GAN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):      
        parent_parser = DeepLabV3plus_Depthcustom.add_model_specific_args(parent_parser)
        
        
        parser = parent_parser.add_argument_group("model")

        
        #parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--b1", type=float, default=.5)
        parser.add_argument("--b2", type=float, default=.999)
        return parent_parser
    
    def __init__(
        self,

        validation_z=None,
        **cfg
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters(cfg)


        
        # networks
 
        self.generator = Generator(**self.hparams)
        self.discriminator = Discriminator()

        self.validation_z = validation_z#torch.randn(8, self.hparams.latent_dim)

        #self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        y_hat = self.generator(z)
        if self.hparams['resize_logits']:
            logits = y_hat['upsampled_logits']        
        else:
            logits = y_hat['logits']

        return logits

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):

        sample = batch
  
        if self.hparams['resize_logits']:
            labels = sample['processed_mask']
        else:
            labels = sample['processed_mask']
            labels = torch.nn.functional.interpolate(labels, logits.size()[2:], mode='bilinear',align_corners=True)


        imgs=batch['processed_image']
        depthmaps = labels

        # sample noise
        z = imgs
     

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)
            logits = y_hat = self.generated_imgs 

 


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

            l1_loss = self.generator.loss_fn(logits,depthmaps)

            tqdm_dict = {"g_loss": g_loss,"l1_loss":l1_loss}
            output = OrderedDict({"loss": g_loss+l1_loss,"y_hat":logits, "progress_bar": tqdm_dict, "log": tqdm_dict})
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

            logits = y_hat= self(z).detach()

     
 
            fake_loss = self.adversarial_loss(self.discriminator(logits).view(-1,1), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict,"y_hat":logits.detach(), "log": tqdm_dict})
            return output

    def validation_step(self, batch, batch_idx):

        sample = batch
        logits = y_hat = self(sample['processed_image'])

        if self.hparams['resize_logits']:

            labels = sample['processed_mask']
        else:

            labels = sample['processed_mask']
            labels = torch.nn.functional.interpolate(labels, logits.size()[2:], mode='bilinear',align_corners=True)
        
        imgs=batch['processed_image']
        depthmaps = labels
        #val_loss = F.binary_cross_entropy_with_logits(y_hat['upsampled_logits'], sample['processed_mask'],weight=sample['weights'])
        val_loss = self.generator.loss_fn(logits, labels)#,weight=sample['weights'])
        
        #if self.hparams['add_classification_head']:
        #    clf_loss = F.binary_cross_entropy_with_logits(y_hat['clf_logits'], sample['label'])
        #    val_loss = val_loss + clf_loss
        #    self.log('{}/val_clfloss'.format(self.hparams['experiment_version']),clf_loss,on_step=False,on_epoch=True,prog_bar=True)
              
        with torch.no_grad():
            metrics = self.generator.valid_metrics(logits, labels)
            

        self.log_dict(metrics,on_step=True, on_epoch=True,prog_bar=False)
        self.log('{}/val_loss'.format(self.hparams['experiment_version']),val_loss,on_step=False,on_epoch=True,prog_bar=True)
        return {'val_loss':val_loss,'y_hat':logits.detach()}

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def __on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.final_seg.conv1.weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


class FineGAN(GAN):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
 
        parser.add_argument("--encoder", type=str, default="resnet50")
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument("--output_stride", type=int, default=16)
        parser.add_argument("--add_laws", type=bool, default=False)
        parser.add_argument("--add_classification_head", type=bool, default=False)
        parser.add_argument("--dropout_p", type=float, default=.9)
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--loss", type=str, default="bce")
        
        parser.add_argument("--l2_reg", type=float, default=5e-4)
        parser.add_argument("--coarsenet_checkpoint", type=str, default='none')
        
        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--b1", type=float, default=.5)
        parser.add_argument("--b2", type=float, default=.999)
        return parent_parser
    
    def __init__(self,
            validation_z=None,
            **cfg):
 

        GAN.__init__(self,       
        validation_z,
        #latent_dim: int = 100,
        **cfg)
        print(self.hparams)

        self.coarse_net = DeepLabV3plus_Depthcustom.load_from_checkpoint(self.hparams['coarsenet_checkpoint']) if isinstance(self.hparams['coarsenet_checkpoint'],str) else  DeepLabV3plus_Depthcustom(**self.hparams)
    def forward(self, z):
        with torch.no_grad():
            z = self.coarse_net(z)['upsampled_logits']
        return self.generator(z)['upsampled_logits']

"""
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
"""


"""
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
"""