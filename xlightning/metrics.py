from torchmetrics import Metric
import torchmetrics
import torch

from typing import Any, Callable, Optional

import torch
from torch import Tensor, tensor

from torchmetrics.metric import Metric
import pytorch_lightning as pl


if torchmetrics.__version__ == '0.11.0':
    from torchmetrics.functional.regression.mse import (
        _mean_squared_error_compute,
        _mean_squared_error_update,
    )

    from torchmetrics.functional.regression.mae import (
        _mean_absolute_error_compute,
        _mean_absolute_error_update,
    )
else:
    from torchmetrics.functional.regression.mean_squared_error import (
        _mean_squared_error_compute,
        _mean_squared_error_update,
    )
    from torchmetrics.functional.regression.mean_absolute_error import (
        _mean_absolute_error_compute,
        _mean_absolute_error_update,
    )
def mask_invalid(preds,target,cutoff=None,**cfg):
    mask = (target < cfg['min_dist']) | (target > cutoff )

    preds=preds[~mask]
    target=target[~mask]

    if cfg['disparity']:
        preds = 1./preds
    
    return preds , target

class BaseMetric(Metric):
    is_differentiable = True
    higher_is_better = False
    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        cfg,
        cutoff: float = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        metrics_cutout: float = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn)
        
        self.cutoff = cutoff if isinstance(cutoff,float) else cfg['max_dist']
        self.cfg = cfg
        self.add_state("sum_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        

    def compute(self) -> Tensor:
        return self.sum_error / self.total

class MeanAbsoluteError(BaseMetric):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)           
        sum_error, n_obs = _mean_absolute_error_update(preds, target)
        self.sum_error += sum_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_error, self.total)


class MeanSquaredError(BaseMetric):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)
        squared_errors = torch.pow(preds-target, 2)
        self.sum_error += torch.sum(squared_errors).item()
        self.total += target.numel()

    def compute(self) -> Tensor:
        return _mean_squared_error_compute(self.sum_error , self.total)


class RootMeanSquaredError(MeanSquaredError):

    def compute(self) -> Tensor:
        return _mean_squared_error_compute(self.sum_error, self.total, squared=False) ## ====> Review


class d1(BaseMetric):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore

        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)  
        thresh = torch.maximum((preds / target), (target / preds))
        _d1 = torch.sum(1.*(thresh < 1.25)).item()
        self.sum_error += _d1 
        self.total += target.numel()


class d2(BaseMetric):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore

        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)
        thresh = torch.maximum((preds / target), (target / preds))
        _d2 = torch.sum(1.*(thresh < 1.25**2)).item()
        self.sum_error += _d2
        self.total += target.numel()


class d3(BaseMetric):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)
        thresh = torch.maximum((preds / target), (target / preds))
        _d3 = torch.sum(1.*(thresh < 1.25**3)).item()
        self.sum_error += _d3
        self.total += target.numel()


class MeanSquaredErrorRel(BaseMetric):
 
    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)
        squared_errors = ((target - preds)**2 )/ target
        self.sum_error += torch.sum(squared_errors).item()
        self.total += target.numel()


class MeanAbsoluteErrorRel(BaseMetric):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)
        sum_abs_error = torch.sum(torch.abs(target - preds) / target   ).item()        
        self.sum_error += sum_abs_error
        self.total += target.numel()


class LogRootMeanSquaredError(RootMeanSquaredError):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)                   
        sum_squared_error = torch.sum((torch.log(preds) - torch.log(target))**2).item()
        n_obs=target.numel()
        self.sum_error += sum_squared_error
        self.total += n_obs

    def compute(self) -> Tensor:
        return torch.sqrt(self.sum_error / self.total)


class LogMeanAbsoluteError(MeanAbsoluteError):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)
        sum_abs_error, n_obs = _mean_absolute_error_update(torch.log(preds), torch.log(target))
        self.sum_error += sum_abs_error
        self.total += n_obs 

    def compute(self) -> Tensor:
        return _mean_absolute_error_compute(self.sum_error, self.total)
        

class SiLog(BaseMetric):
  
    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)      
        err = (torch.log(preds) - torch.log(target))
        sum_squared_rel_error = (torch.mean(err**2) - torch.mean(err)**2).item()
        self.sum_error += sum_squared_rel_error
        self.total += 1

    def compute(self) -> Tensor:
        return torch.sqrt(self.sum_error/self.total)*100
 
class log10(BaseMetric):
 
    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds , target = mask_invalid(preds,target,cutoff=self.cutoff,**self.cfg)
        err = torch.abs(torch.log10(preds) - torch.log10(target))
        sum_squared_rel_error = torch.sum(err).item()
        self.sum_error += sum_squared_rel_error
        self.total += target.numel()



