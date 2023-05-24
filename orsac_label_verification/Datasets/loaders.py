import os

import pandas as pd
from fastai.data.load import DataLoader as FastDataLoader
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from orsac_label_verification.utils.logging import (
    current_iter_path,
    get_data_csv,
    get_img_abspath,
)
from orsac_label_verification.utils.utils import (
    alb_transform_test,
    alb_transform_train,
)


def get_test_loader(config, mode,split="Test", test_df=None):
    """sets up the torch data loaders for testing"""
    if test_df is None and mode!='Eval':
        df = pd.read_csv(os.path.join(current_iter_path(config), "data.csv"))

        df = df[
            df.Id.apply(lambda x: os.path.exists(get_img_abspath(x, config)))
        ].reset_index(drop=True)
        test_df = df[df.Split == split].reset_index(drop=True)
    elif config.test_data_csv_path is not None and mode =='Eval':
        df = pd.read_csv(get_data_csv(config.test_data_csv_path))
        df = df[
            df.Id.apply(lambda x: os.path.exists(get_img_abspath(x, config)))
        ].reset_index(drop=True)
        test_df = df[df.Split == split].reset_index(drop=True)

    # set up the datasets
    DatasetClass = config.get_test_dataset()
    test_dataset = DatasetClass(
        config, test_df, transformer=alb_transform_test(config.imsize)
    )

    # set up the data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader


def get_fastai_dataloaders(config, train_df=None, valid_df=None, train_sampler=None):
    DatasetClass = config.get_train_dataset()
    if train_df is None:
        df = pd.read_csv(os.path.join(current_iter_path(config), "data.csv"))

        df = df[
            df.Id.apply(lambda x: os.path.exists(get_img_abspath(x, config)))
        ].reset_index(drop=True)
        train_df = df[df.Split == "Train"].reset_index(drop=True)
        if valid_df is None:
            valid_df = df[df.Split == "Valid"].reset_index(drop=True)

    train_ds = DatasetClass(
        config, train_df, transformer=alb_transform_train(config.imsize)
    )
    valid_ds = DatasetClass(
        config, valid_df, transformer=alb_transform_train(config.imsize)
    )
    if config.sampling == "basic" or config.sampling == "oversampling":
        train_dl = FastDataLoader(
            train_ds,
            batch_size=config.batch_size,
            batch_sampler=train_sampler,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        valid_dl = FastDataLoader(
            valid_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    elif config.sampling == "random":
        train_dl = FastDataLoader(
            train_ds,
            batch_sampler=ImbalancedDatasetSampler(train_ds),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        valid_dl = FastDataLoader(
            valid_ds,
            batch_sampler=ImbalancedDatasetSampler(valid_ds),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return [train_dl, valid_dl]
