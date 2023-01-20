
def override_cfg(cfg,global_cfg):
    with open(global_cfg,'r') as f:
        gcfg = f.readlines()
        gcfg = [i.strip('\n') for i in gcfg]
    for g in gcfg:
        k,v = g.split('=')
        cfg[k]=v
        print(k,v)
    return cfg

class ConfigParser:
    def __init__(self,file):
        with open(file,'r') as f:
            self.config = f.readlines()
            self.config = [i.strip('\n') for i in self.config]
            self.config = {i.split('=')[0]:i.split('=')[1] for i in self.config }

    def override(self,cfg):
        for k,v in self.config.items():
            cfg[k] = type(cfg[k])(self.config[k])
        return cfg






import numpy as np
import itertools
class GenerateExperiment:
    def __init__(self,num_folds=14):
        folds = np.arange(1,num_folds+1).tolist() 
        models = ['glp','newcrf','adabins','bts','dpt_large','dpt_hybrid']
        self.experiments = list(itertools.product(models, folds))
        pass


        