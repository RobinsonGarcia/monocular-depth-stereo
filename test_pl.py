#import pytorch_lightning as pl
from pytorch_lightning import Trainer
import xlightning.models as xlm
from xlightning.datamodules import DepthDataModule
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import argparse
import json
import os
#import wandb
import sys

from pytorch_lightning.callbacks import StochasticWeightAveraging

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        print('dir already exists')
    pass


if __name__=="__main__":
    #python train_pl.py --lr=1e-5 --disparity=false --extend_3d=false --extended=true --l2_reg=0 --loss=scale --max_dist=150 --min_dist=.02 --nearest_up=true  --accumulate_grad_batches=6 --mask_background=false --gpus=4 --strategy=ddp --pretrained_kitti=true --pred_log=false --resize_H_to=384 --SIZE 384 --batch_size=1#
 
    experiments=[('newcrf', 1),
    ('bts', 1),
    ('dpt_hybrid', 1),
    ('newcrf', 2),
    ('bts', 2),
    ('dpt_hybrid', 2),
    ('newcrf', 3),
    ('bts', 3),
    ('dpt_hybrid', 3),
    ('newcrf', 4),
    ('bts', 4),
    ('dpt_hybrid', 4),
    ('newcrf', 5),
    ('bts', 5),
    ('dpt_hybrid', 5),
    ('newcrf', 6),
    ('bts', 6),
    ('dpt_hybrid', 6),
    ('newcrf', 7),
    ('bts', 7),
    ('dpt_hybrid', 7),
    ('newcrf', 8),
    ('bts', 8),
    ('dpt_hybrid', 8),
    ('newcrf', 9),
    ('bts', 9),
    ('dpt_hybrid', 9),
    ('newcrf', 10),
    ('bts', 10),
    ('dpt_hybrid', 10),
    ('newcrf', 11),
    ('bts', 11),
    ('dpt_hybrid', 11),
    ('newcrf', 12),
    ('bts', 12),
    ('dpt_hybrid', 12),
    ('newcrf', 13),
    ('bts', 13),
    ('dpt_hybrid', 13)]


    TASK_ID = 2 #int(os.environ['SLURM_ARRAY_TASK_ID'])
    MODEL_NAME , FOLD = experiments[TASK_ID]

    WANDB_PROJECT='mde_highres_kfold'

    print(MODEL_NAME , FOLD,TASK_ID)

    parser = argparse.ArgumentParser()


    parser = argparse.ArgumentParser(
            description="pytorch lightning + classy vision TorchX example app"
        )


    print('parsing args...')

    LitModel = xlm.model_factory(**{'model':MODEL_NAME})

    DataModule = DepthDataModule

    parser = LitModel.add_model_specific_args(parser)

    parser = DataModule.add_model_specific_args(parser)

    parser.add_argument(
            "--accelerator", type=str, default='cuda', help="."
        )

    parser.add_argument(
            "--strategy", type=str, default='ddp', help="."
        )

    parser.add_argument(
            "--gpus", type=int, default=4, help="."
        ) 

    parser.add_argument(
            "--num_nodes", type=int, default=1, help="."
        ) 

    parser.add_argument(
            "--sync_batchnorm",action='store_true'
        ) 

    parser.add_argument(
            "--accumulate_grad_batches", type=int, default=2, help="."
        )  

    parser.add_argument(
            "--max_epochs", type=int, default=500, help="."
        )  

    parser.add_argument(
            "--cmap", type=str, default='plasma', help="sets the depthmap colormap"
        ) 

    parser.add_argument(
            "--gan", type=str, default='false', help="sets the depthmap colormap"
        ) 

    cfg = vars(parser.parse_args())

    cfg['model']=MODEL_NAME
    cfg['fold']=FOLD

    if (MODEL_NAME=='adabins') or (MODEL_NAME=='dpt_large'):
        cfg['freeze_encoder']=True
    #=== Logger:

    cfg['experiment_version']=f'{MODEL_NAME}'
    mkdir('lightning_logs/depth')
    mkdir('lightning_logs/depth/{}'.format(cfg['experiment_version']))
    mkdir('lightning_logs/depth/{}/wandb'.format(cfg['experiment_version']))

    cfg['WANDB_PROJECT'] = WANDB_PROJECT
    cfg['MODEL_NAME'] = MODEL_NAME 
    cfg['TASK_ID'] = TASK_ID 

    WANDB_PROJECT = cfg['WANDB_PROJECT']
    MODEL_NAME = cfg['MODEL_NAME']
    TASK_ID = cfg['TASK_ID']

    csv_logger = CSVLogger("lightning_logs", name="depth",version=cfg['experiment_version'])
    tb_logger = TensorBoardLogger("lightning_logs", name='depth', version=cfg['experiment_version'], log_graph=False, default_hp_metric=False, prefix='', sub_dir=None)
    cfg['experiment_version'] = tb_logger.version
    cfg['monitor']='{}/val_loss'.format(tb_logger.version)
    cfg['{}/val_loss'.format(tb_logger.version)]=-1
    cfg['depthnet']=True
    

    logger = []#WandbLogger(project=WANDB_PROJECT,name=f'{MODEL_NAME}_{TASK_ID}',log_model=True,reinit=True,group=None,save_dir='lightning_logs/depth/{}'.format(cfg['experiment_version']))]
    
    os.environ["WANDB_DIR"] = 'lightning_logs/depth/{}'.format(cfg['experiment_version'])

    #=== GAN WRAPPER: 
    # LitGan = xlm.model_factory(**{'model':'GAN'})(generator=model)
    #=== GAN WRAPPER: 
    print('loading model...')
    if cfg['gan']=='true':
        from xlightning.models.depth.gans_wrapper import GAN
        model = GAN(generator=LitModel(**cfg),**cfg)
    else:
        model = LitModel(**cfg)

    #logger[0].watch(model,log="all")
    #=== load datamodule:
    print('loading datamodule...')
    dm = DataModule(**cfg)
    #dm.setup(mode='train')

    #=== load callbacks:
    print('loading callbacks...')
    from xlightning.callbacks import load_callbacks

    cfg['checkpoint_path']=f'lightning_logs/{WANDB_PROJECT}/{MODEL_NAME}_{TASK_ID}/checkpoints'
    cfg['monitor']='{}/valid_mae_5'.format(cfg['experiment_version'])
    callbacks=load_callbacks(**cfg)+ [StochasticWeightAveraging(swa_lrs=1e-2)]


    for p in model.model.parameters():
        p.requires_grad = False
    model.norm_layer.requires_grad=True
    model.norm_layer.weight.data.fill_(0.00006016,)
    model.norm_layer.bias.data.fill_(0)

    #=== trainer
    print(cfg)
  
    trainer = Trainer(
        devices=1,
        #gpus=8,
        num_nodes=1,
        #strategy='ddp',
        accelerator='auto',
        auto_lr_find=False,
        max_epochs=cfg['max_epochs'],
        logger=logger,
        callbacks=callbacks,
        sync_batchnorm=cfg['sync_batchnorm'],
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg['accumulate_grad_batches'],
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value")

    trainer.fit(model,dm)