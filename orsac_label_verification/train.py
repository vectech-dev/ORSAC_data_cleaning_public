import json
import os
from functools import partial

from dotenv import load_dotenv

load_dotenv()

from fastai.callback.schedule import fit_one_cycle
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.optimizer import ranger
import torch
from orsac_label_verification.Datasets.loaders import get_fastai_dataloaders
from orsac_label_verification.test import test
from orsac_label_verification.train_config import ExperimentationConfig
from orsac_label_verification.utils.callbacks import SaveBestModel
from orsac_label_verification.utils.logging import (
    con_mat_iter,
    create_results_df,
    create_top_preds_df,
    experiment_path,
    init_data_df,
    init_metrics_df,
    iter_data,
    log_results,
    models_directory,
    save_current_iter,
    save_test_output,
    setup_directories,
    shuffle_data,
)
from orsac_label_verification.utils.metrics import accuracy, macro_f1
from orsac_label_verification.utils.utils import SimpleLogger


def train_(config, model):
    opt_func = partial(
        ranger, mom=getattr(config, "mom", 0.9), eps=getattr(config, "eps", 1e-5)
    )
    metrics = [
        partial(accuracy),
        partial(macro_f1),
    ]
    best_save_cb = SaveBestModel(config=config)
    cbs = [best_save_cb]
    if config.Early_Stopping and config.early_stopping_patience >= 0:
        cbs.append(
            EarlyStoppingCallback(
                monitor="valid_loss",
                min_delta=0.001,
                patience=config.early_stopping_patience,
            )
        )
    loss_function = config.get_loss()
    loss = loss_function(**config.loss_kwargs)
    data = DataLoaders(*get_fastai_dataloaders(config))
    learn = Learner(
        data,
        model,
        loss_func=loss,
        opt_func=opt_func,
        lr=config.lr,
        metrics=metrics,
        wd=config.weight_decay,
        model_dir=models_directory(config),
    )
    learn.to(config.device)
    learn.unfreeze()
    learn.fit_one_cycle(config.epochs, cbs=cbs)
    return learn.model


def orsac_one_iter(config, i):
    
    print(f"Running iter_{i}")
    Model = config.get_model()
    kwargs = config.model_kwargs
    model = Model(**kwargs)
    device = config.device
    model = model.to(device)
    if config.device_ids is not None:
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    model.train()
    iter_data(config, i)
    train_(config, model)
    probs, preds, labels, acc = test(config)
    con_mat_iter(config, labels, preds, i)
    save_test_output(config, preds, probs, acc)
    log_results(config)

    # Skip this, consider altogether removing top_x 
    # record_top_x_pred(config)
    save_current_iter(config, i)


def orsac_all(config):
    #Another place where time is wasted. Instead, in orsac init have a split column added. OR, just call this shuffle data function if there is not a split column.

    shuffle_data(config)
    for i in range(config.n_iterations):
        orsac_one_iter(config, i)


def init_orsac(config):
    setup_directories(config)
    init_data_df(config)
    init_metrics_df(config)
    create_results_df(config)
    with open(os.path.join(experiment_path(config), "config.json"), "w") as outfile:
        json.dump(config.dict(exclude_none=True), outfile)


def train_model(config):
    init_orsac(config)
    orsac_all(config)


def train(config: ExperimentationConfig):
    print("Training...")
    assert config.mode == "train", "Incorrect settings"
    print("configs are good...")
    logger = SimpleLogger(config.model_name + "-Trainer")
    print("logger created...")
    if os.path.exists(f"experiments/{config.exp_name}"):
        raise Exception(f"Error: exp'{config.exp_name}' already exists.")

    logger.log("Training Settings Dump: ")
    print(json.dumps(config.dict(exclude_none=True), indent=2))

    
    train_model(config)


def train_eval(config: ExperimentationConfig):
    Model = config.get_model()
    kwargs = config.model_kwargs
    model = Model(**kwargs)
    device = config.device
    model = model.to(device)
    if config.device_ids is not None:
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    model.train()
    train_(config, model)
