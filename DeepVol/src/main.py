import os
import sys
import math
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from pl_bolts.datamodules import SklearnDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from metrics import MetricsCalculator
from deepvol import DeepVol
from args import get_args

logging.captureWarnings(True)

class DeepVolModule(pl.LightningModule):

    def __init__(self, hparams, loss_fn=nn.MSELoss(), log_grads: bool = False, use_sentence_split: bool = True):

        super().__init__()
        self.exec_identifier = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace('/',':').replace(' ', '_')

        self.use_sentence_split = use_sentence_split
        self.log_grads = log_grads

        self.conditioning_range = hparams.conditioning_range
        self.save_dataset_heavy_format = hparams.save_dataset_heavy_format
        self.granularity = hparams.granularity
        self.optimizer = hparams.optimizer
        self.output_length = hparams.out_len
        self.win_len = hparams.win_len
        self.optimizer = hparams.optimizer
        symbols = self._get_available_tickers()
        self._setup_dataloaders()
        self.loss_fn = loss_fn
        self.lr = hparams.lr

        self.metrics = MetricsCalculator(["mean_absolute_error", "mean_squared_error", "symmetric_mean_absolute_percentage_error", "qlike", "r2_score", "max_error", "mean_squared_log_error", "median_absolute_error","mean_squared_error_not_root"]) 
        self.model = DeepVol(num_blocks=hparams.num_blocks, num_layers=hparams.num_layers, num_classes=1,
                             output_len=self.output_length, ch_start=1,
                             ch_residual=hparams.ch_residual, ch_dilation=hparams.ch_dilation, ch_skip=hparams.ch_skip,
                             ch_end=hparams.ch_end, kernel_size=hparams.kernel_size, bias=True)
        self.ReLU = nn.ReLU()

    def qlike(self, h, data): 
        K = data.shape[0] 
        likConst = K*math.log(2 * math.pi)
        ll = torch.sum(0.5 * (likConst + torch.sum(torch.log(h + torch.tensor(1e-1 )),axis=0) + torch.sum(torch.divide(data, h + torch.tensor(1e-1 )), axis=0)))
        return ll

    def forward(self, x):
        return self.model(x) 

    def _forward_batch(self, batch):
        x, y = batch
        x = x.float()
        x = x[:][:, 3:]
        y_hat = self.ReLU(self.forward(x.unsqueeze(1)).squeeze(-1))         
        loss = self.qlike(y_hat.T, y.T)                                    
        return loss, y, y_hat                                     
        
    def training_step(self, batch, batch_idx):
        loss, true, preds = self._forward_batch(batch)[:3]
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        dict_train = self.metrics.generate_logs(loss, preds, true, "train")
        for key, value in dict_train.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"loss": loss, "log": dict_train}

    def training_step_end(self, training_out):
        tensorboard_logs = self.metrics.generate_mean_metrics(training_out["log"], "train")
        for key, value in tensorboard_logs.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"loss": training_out["loss"], "progress_bar": tensorboard_logs, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, true, preds = self._forward_batch(batch)[:3]
        return self.metrics.generate_logs(loss, preds, true, "val")

    def validation_epoch_end(self, outputs):
        tensorboard_logs = self.metrics.generate_mean_metrics(outputs, "val")
        for key, value in tensorboard_logs.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"progress_bar": tensorboard_logs, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, true, preds = self._forward_batch(batch)[:3]
        dict_test = self.metrics.generate_logs(loss, preds, true, "test")
        for key, value in dict_test.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return dict_test

    def test_epoch_end(self, outputs):
        tensorboard_logs = self.metrics.generate_mean_metrics(outputs, "test")
        for key, value in tensorboard_logs.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"progress_bar": tensorboard_logs, "log": tensorboard_logs}
    

    def configure_optimizers(self):
        if(self.optimizer == "LBFGS"): #Notice that the optimization algorithm LBFGS cannot be used with multiples GPUs
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr)
        elif(self.optimizer == "ASGD" or self.optimizer == "SGD"):
            optimizer = torch.optim.ASGD(self.parameters(), lr=self.lr)
        elif(self.optimizer == "Adam"):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            sys.exit("Could not save the dataset in HEAVY format")
        return optimizer

    def _get_available_tickers(self):
        """
        Returns a list of available tickers for the given granularity.
        """
        pass


    def _setup_dataloaders(self):
        """
        This function must set up the dataloaders for training, validation, and test datasets.
        Depending on the data and the experiment, different implementations should handle 
        splitting by dates and by tickers.
        """

        """
        dataset_train = SklearnDataset(x_train, y_train)
        loader_train = DataLoader(dataset_train) 

        dataset_val = SklearnDataset(x_val, y_val)
        loader_val = DataLoader(dataset_val)

        dataset_test = SklearnDataset(x_test, y_test)
        loader_test = DataLoader(dataset_test)

        self.dl_train = loader_train
        self.dl_valid = loader_val
        self.dl_test = loader_test
        """
        pass

    
    @property
    def num_classes(self):
        return self.dl_train.dataset.num_chars
        #return 1

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_valid

    def test_dataloader(self):
        return self.dl_test

    def on_train_start(self):

        input = torch.ones((self.batch_size_train, 1, self.win_len))
        if torch.cuda.is_available():
            input = input.cuda()

        self.logger.experiment.add_graph(self.model, input)

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.log_grads and self.trainer.global_step % 100 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(tag=name, values=grads,
                                                     global_step=self.trainer.global_step)

from args import get_args
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":

    # argument parsing
    args = get_args()

    args.patience = 100
    args.n_epochs = 1000
    args.check_val_every_n_epoch = 50
    args.num_blocks = 2
    args.num_layers = 2
    args.kernel_size =5
    args.granularity = "5min"

    model = DeepVolModule(args, use_sentence_split=True) 

    parameters_summary = str(model.exec_identifier)
    logger = TensorBoardLogger("logs", name="deepvol", version=parameters_summary)

    checkpoint_callback = ModelCheckpoint(monitor="val_mean_squared_error", save_top_k=1)
    ignore_multidevice= (model.optimizer == "LBFGS") #LBFGS optimization algorithm cannot work on >1 gpus

    if (ignore_multidevice):
        trainer = pl.Trainer(logger=logger, max_epochs=10000, gpus=[0] if torch.cuda.is_available() else None, check_val_every_n_epoch=args.check_val_every_n_epoch, callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=50)]) 
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.n_epochs, gpus=[0] if torch.cuda.is_available() else None, check_val_every_n_epoch=args.check_val_every_n_epoch, callbacks=[checkpoint_callback, EarlyStopping(monitor="val_mean_absolute_error", mode="min", patience=args.patience)]) 

    trainer.fit(model) 
