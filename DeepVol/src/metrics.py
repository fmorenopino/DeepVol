from pdb import set_trace
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error, max_error, mean_squared_log_error, r2_score
import math
import numpy as np
class MetricsCalculator(object):

    def __init__(self, metrics) -> None:
        super().__init__()
        self.metrics = metrics

    def _format_tensor(self, tensor):
        return tensor.cpu().detach().view(-1, )

    def _mean_metrics(self, logs, key):
        return logs[key] if isinstance(logs, dict) else torch.stack([item[key] for item in logs]).mean()

    def accuracy(self, true, pred):
        return torch.tensor(accuracy_score(self._format_tensor(true), self._format_tensor(pred)))

    def precision(self, true, pred):
        return torch.tensor(precision_score(self._format_tensor(true), self._format_tensor(pred), average="macro"))

    def recall(self, true, pred):
        return torch.tensor(recall_score(self._format_tensor(true), self._format_tensor(pred), average="macro"))

    def f1(self, true, pred):
        return torch.tensor(f1_score(self._format_tensor(true), self._format_tensor(pred), average="macro"))

    def mean_absolute_error(self, true, pred):
        return torch.tensor(mean_absolute_error(self._format_tensor(true), self._format_tensor(pred)))
    
    def mean_absolute_percentage_error(self, true, pred):
        return torch.tensor(mean_absolute_percentage_error(self._format_tensor(true), self._format_tensor(pred)))

    def mean_squared_error(self, true, pred):
        return torch.tensor(mean_squared_error(self._format_tensor(true), self._format_tensor(pred), squared=False))
    
    def mean_squared_error_not_root(self, true, pred):
        return torch.tensor(mean_squared_error(self._format_tensor(true), self._format_tensor(pred)))

    def symmetric_mean_absolute_percentage_error(self, true, pred):
        return torch.tensor(0.5*torch.mean(torch.abs(true-pred)/(abs(true)+abs(pred) + 1e-6)))

    def r2_score(self, true, pred):
        return torch.tensor(r2_score(self._format_tensor(true), self._format_tensor(pred)))

    def max_error(self, true, pred):
        return torch.tensor(max_error(self._format_tensor(true), self._format_tensor(pred)))

    def mean_squared_log_error(self, true, pred):
        return torch.tensor(max_error(self._format_tensor(true), self._format_tensor(pred)))

    def median_absolute_error(self, true, pred):
        return torch.tensor(median_absolute_error(self._format_tensor(true), self._format_tensor(pred)))

    def qlike(self, true, pred):
        true = true.T 
        pred = pred.T
        K = true.shape[0] 
        likConst = K*math.log(2 * math.pi)
        cte = 1e-1 
        ll = torch.sum(0.5 * (likConst + torch.sum(torch.log(pred + torch.tensor(cte)),axis=0) + torch.sum(torch.divide(true, pred + torch.tensor(cte)), axis=0)))

        return ll

    def generate_logs(self, loss, preds, true, prefix):
        return {f"{prefix}_loss": loss,
                **{f"{prefix}_{key}": MetricsCalculator.__dict__[key](self, true, preds) for key in self.metrics}}

    def generate_mean_metrics(self, outputs, prefix):
        mean_keys = ["loss"] + self.metrics
        return {f"{prefix}_{key}": self._mean_metrics(outputs, f"{prefix}_{key}") for key in mean_keys}