import argparse
from xlightning.models.pl_adabins import AdaBins
from xlightning.models.pl_bts import BTS
from xlightning.models.pl_dpt import DPTmodel
from xlightning.models.pl_newcrf import NewCRF
from xlightning.models.pl_pixelformer import PIXelFormer
import torch.nn as n

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
      
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)


MODEL_CLASSES = [AdaBins,BTS,DPTmodel,NewCRF,PIXelFormer]
MODEL_CLASS = DPTmodel

parser = argparse.ArgumentParser()

parser = MODEL_CLASS.add_model_specific_args(parser)

args = parser.parse_args()

cfg = vars(args)

cfg['experiment_version']=1

model = MODEL_CLASS(**cfg)