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

from ransac_label_verification.Datasets.loaders import get_fastai_dataloaders
from ransac_label_verification.test import test
from ransac_label_verification.train_config import ExperimentationConfig
from ransac_label_verification.utils.callbacks import SaveBestModel
from ransac_label_verification.utils.logging import (
    con_mat_iter,
    create_results_df,
    create_top_preds_df,
    experiment_path,
    init_data_df,
    init_metrics_df,
    iter_data,
    log_results,
    models_directory,
    record_top_x_pred,
    save_current_iter,
    save_test_output,
    setup_directories,
    shuffle_data,
)
from ransac_label_verification.utils.metrics import accuracy, macro_f1
from ransac_label_verification.utils.utils import SimpleLogger


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


def ransac_one_iter(config, model, i):
    print(f"Running iter_{i}")
    iter_data(config, i)
    train_(config, model)
    probs, preds, labels, acc = test(config)
    con_mat_iter(config, labels, preds, i)
    save_test_output(config, preds, probs, acc)
    log_results(config)
    record_top_x_pred(config)
    save_current_iter(config, i)


def ransac_all(config, model):
    shuffle_data(config)
    for i in range(config.n_iterations):
        ransac_one_iter(config, model, i)


def init_ransac(config):
    setup_directories(config)
    init_data_df(config)
    init_metrics_df(config)
    create_results_df(config)
    create_top_preds_df(config)
    with open(os.path.join(experiment_path(config), "config.json"), "w") as outfile:
        json.dump(config.dict(exclude_none=True), outfile)


def train_model(config, model):
    init_ransac(config)
    ransac_all(config, model)


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

    Model = config.get_model()
    logger.log(f"Using model {config.model_name}")
    kwargs = config.model_kwargs
    model = Model(**kwargs)
    device = config.device
    model = model.to(device)
    model.train()
    train_model(config, model)


def train_eval(config: ExperimentationConfig):
    Model = config.get_model()
    kwargs = config.model_kwargs
    model = Model(**kwargs)
    device = config.device
    model = model.to(device)
    model.train()
    train_model(config, model)
