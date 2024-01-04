import os
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type

import torch.utils.data
from pretrainedmodels import xception
from pydantic import BaseModel, Field, ValidationError, validator
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.utils.data import Dataset

from orsac_label_verification.Datasets.datasets import MosDataset
from orsac_label_verification.models.models import efficientnet_mod, xception_mod
from orsac_label_verification.utils.losses import MosLoss
from orsac_label_verification.utils.utils import get_classes

from orsac_label_verification.utils.logging import get_data_csv

# todo but not high priority
class ExperimentationConfig(BaseModel):
    """The training/testing configuration."""

    models: ClassVar[Dict[str, torch.nn.Module]] = {
        "Xception": xception_mod,
        "og_xception": xception,
        "Efficientnet": efficientnet_mod,
    }

    datasets: ClassVar[Dict[str, Dataset]] = {
        "MosDataset": MosDataset,
    }

    evaluation_metrics: ClassVar[Dict[str, Callable]] = {}

    optimizers: ClassVar[Dict[str, Type]] = {
        "AdamW": AdamW,
        "SGD": SGD,
        "RMSProp": RMSprop,
        "Adam": Adam,
    }

    losses: ClassVar[Dict[str, Type]] = {
        "L1Loss": L1Loss,
        "MosLoss": MosLoss,
        "CrossEntropy": CrossEntropyLoss,
    }
    class_names: Optional[list] = Field(
        default={}, description="the names of the classes, gotten automatically"
    )
    n_iterations: Optional[int] = Field(
        default=35, description="the number of iterations to run ORSAC for"
    )
    weight_decay: Optional[float] = Field(default=1e-2, description="weight decay rate")
    lr: Optional[float] = Field(default=1e-3, description="learning rate")
    mom: Optional[float] = Field(default=0.9, description="ranger optimizer parameter")
    eps: Optional[float] = Field(default=1e-6, description="ranger optimizer parameter")
    loss_dict: Optional[dict] = Field(
        default={"FocalLoss": {"weight": 1.0, "mag_scale": 1.0, "gamma": 2}}
    )
    device: Optional[str] = Field(
        default="cuda", description="the device to send the model and data to"
    )
    imsize: Optional[int] = Field(default=299, description="the size of the image")
    model_name: str = Field(..., description="The model to train/test with")
    exp_name: str = Field(
        ..., description="Experiment Name (and where tofind in experiments folder)"
    )
    data_csv_path: str = Field(..., description="Path to the dataset df")
    test_data_csv_path: Optional[str] = Field(
        default=None,
        description="Optional separate test set. Only use if you want to use a differenet sheet as your test set.",
    )
    num_classes: Optional[int] = Field(default=0, description="the number of classes")
    top_x: Optional[int] = Field(default=3, description="Parameter for visualizations")
    base_directory: Optional[str] = Field(
        default=None,
        description="The base image directory that will be referenced in all dataframes used in training and testing.",
    )
    best_weights: Optional[str] = Field(
        default="acc",
        description="Which metric to use for the best weights. Options include acc,f1,and loss",
    )
    white_balance: Optional[bool] =Field(
        default=False,
        description='Whether to apply the white balance transform to images or not.'
    )
    mode: str = Field(
        default="clean",
        description="The network mode i.e. `train` or `test` or `finetune`",
    )
    early_stopping_patience: Optional[int] = Field(
        default=0, description="Early Stopping patience"
    )

    epochs: Optional[int] = Field(
        default=30, description="The number of epochs when training"
    )

    Early_Stopping: Optional[bool] = Field(
        default=False,
        description="early stopping to stop the training when the model starts to overfit to the training data.",
    )
    preload_data: Optional[bool] = Field(
        default=False,
        description="whether to preload the data or not",
    )
    batch_size: int = Field(default=64, description="The batch size when training")

    model_kwargs: Dict[str, Any] = Field(
        ..., description="The keyword arguments to the model"
    )

    train_set_name: str = Field(..., description="The training set")

    train_set_kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="The keyword arguments for the training set"
    )

    valid_set_name: Optional[str] = Field(
        default=None, description="The validation set"
    )

    valid_set_kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="The keyword arguments for the validation set"
    )

    test_set_name: Optional[str] = Field(default=None, description="The test set")

    test_set_kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="The keyword arguments for the test set"
    )

    loss: str = Field(default="", description="The loss to use")

    loss_kwargs: Optional[Dict] = Field(
        default={}, description="The loss keyword arguments"
    )
    sampling: Optional[str] = Field(
        default="random",
        description="The sampling method for the dataloader to use.",
    )

    save_dir: Optional[str] = Field(
        default=None,
        description="The directory for the saved models while training",
    )

    weights_path: Optional[str] = Field(
        default=None,
        description="Where to load the model weights from while testing",
    )
    device_ids: Optional[list] = Field(
        default=[0,1,2,3],
        description='gpu id'
    )

    num_workers: Optional[int] = Field(
        default=40,
        description="The number of workers to use in dataloaders",
    )

    test_metrics: Optional[List[str]] = Field(
        default=None, description="The test/evaluation metrics"
    )

    def get_model(self) -> torch.nn.Module:
        return self.models[self.model_name]

    def get_train_dataset(self) -> torch.utils.data.Dataset:
        return self.datasets[self.train_set_name]

    def get_valid_dataset(self) -> torch.utils.data.Dataset:
        if self.valid_set_name:
            return self.datasets[self.valid_set_name]
        else:
            raise AttributeError("No validation set name provided.")

    def get_test_dataset(self) -> torch.utils.data.Dataset:
        if self.test_set_name:
            return self.datasets[self.test_set_name]
        else:
            raise AttributeError("No test set name provided.")

    def get_test_metrics(self) -> Dict[str, Callable]:
        return {metric: self.evaluation_metrics[metric] for metric in self.test_metrics}

    def get_loss(self) -> Callable:
        return self.losses[self.loss]
    def get_class_names(self)->List:
        d1,d2=get_classes(get_data_csv(self.data_csv_path))
        print(d1,d2,'HERE',d1.values)
        return d1.values()

    class Config:
        arbitrary_types_allowed = True
        allow_extra = False
        allow_mutation = False
