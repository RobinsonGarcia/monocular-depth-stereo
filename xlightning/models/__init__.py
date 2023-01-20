from xlightning.models.pl_dpt import DPTmodel
from xlightning.models.pl_newcrf import NewCRF

def model_factory(**cfg):
    if 'dpt' in cfg['model']:
        return DPTmodel
    elif 'crf' in cfg['model']:
        return NewCRF
    else:
        raise
