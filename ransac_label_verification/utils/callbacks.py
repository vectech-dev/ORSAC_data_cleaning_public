# Owned by Johns Hopkins University, created prior to 5/28/2020
import datetime
import os

import numpy as np
import torch
from fastai.callback.tracker import Callback, TrackerCallback
from fastcore.basics import store_attr
from fastcore.nb_imports import *
from torch.utils.data import BatchSampler, WeightedRandomSampler

from ransac_label_verification.utils.logging import save_metric


class SaveBestModel(TrackerCallback):
    _only_train_loop = True

    def __init__(
        self,
        config,
        monitor="valid_loss",
        comp=None,
        min_delta=0.0,
        with_opt=False,
        reset_on_fit=True,
        outfile="",
    ):
        super().__init__(
            monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit
        )
        self.last_saved_path = None
        store_attr("with_opt")
        self.config = config
        self.best_loss = None
        self.best_acc = None
        self.best_f1 = None
        self.metrics_pos_dict = None
        self.outfile = outfile

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def before_fit(self):
        super().before_fit()
        # Metrics - ['epoch', 'train_loss', 'valid_loss', 'accuracy', 'macro_f1', 'time']
        self.metrics_pos_dict = {
            metric: ndx - 1 for ndx, metric in enumerate(self.recorder.metric_names)
        }

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."

        loss = self.recorder.values[-1][self.metrics_pos_dict["valid_loss"]]
        accuracy = self.recorder.values[-1][self.metrics_pos_dict["accuracy"]]
        f1_score = self.recorder.values[-1][self.metrics_pos_dict["macro_f1"]]

        if self.best_acc is None or accuracy > self.best_acc:
            self.best_acc = accuracy
            self._save(f"best_acc" + self.outfile)
        if self.best_f1 is None or f1_score > self.best_f1:
            self.best_f1 = f1_score
            self._save(f"best_f1" + self.outfile)
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self._save(f"best_loss" + self.outfile)

    def after_fit(self):
        metric_used = self.config.best_weights
        if metric_used == "acc":
            metric = self.best_acc
        elif metric_used == "f1":
            metric = self.best_f1
        elif metric_used == "loss":
            metric = self.best_loss
        else:
            metric = self.best_acc
        save_metric(self.config, metric, f"Validation_{self.config.best_weights}")
