
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



#============ GAN ================#



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = timm.create_model('vgg16',features_only=True,pretrained=True, in_chans=1)
        self.sigmoid = torch.sigmoid
    def forward(self, input):
        y = self.encoder(input.float())[-1]
        return self.sigmoid(y)

class GAN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):      
        parent_parser = parent_parser.add_argument_group("model")
        
        
        parser = parent_parser.add_argument_group("model")

        
        #parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--b1", type=float, default=.5)
        parser.add_argument("--b2", type=float, default=.999)
        return parent_parser
    
    def __init__(
        self,
        generator,
        validation_z=None,
        **cfg
    ):
        super().__init__()
        #self.save_hyperparameters()
        self.save_hyperparameters(cfg)

        self.hparams['b1'] = .5
        self.hparams['b2'] = .999


        
        # networks
 
        self.generator = generator
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
            labels = torch.nn.functional.interpolate(labels, logits.size()[2:], mode='nearest')#,align_corners=True)


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

            #l1_loss = self.generator.loss_fn(logits,depthmaps)['loss'] #====> Removed for the first trials

            tqdm_dict = {"g_loss": g_loss}#,"l1_loss":l1_loss}
            output = OrderedDict({"loss": g_loss,"y_hat":logits, "progress_bar": tqdm_dict, "log": tqdm_dict})
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

            logits = y_hat= self(z)#.detach()

     
 
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
            

        self.log_dict(metrics,on_step=False, on_epoch=True,prog_bar=False)
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
