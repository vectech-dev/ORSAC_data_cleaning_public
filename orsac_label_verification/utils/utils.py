import os
from os import makedirs

import numpy as np
import pandas as pd
from albumentations import (
    Compose,
    Flip,
    GaussNoise,
    Normalize,
    RandomRotate90,
    Resize,
    ShiftScaleRotate,
    Transpose,
)
from albumentations.augmentations.transforms import ColorJitter
from PIL import Image


class SimpleLogger:
    def __init__(self, name):
        self.name = name

    def log(self, data):
        print(f"{self.name} - {data}")


def make_dirs(dirs):
    for dir_ in dirs:
        makedirs(dir_, exist_ok=True)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def str_to_list(string):
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.split(",")
    string = [int(i) for i in string]
    return string


def get_classes(datasheet, only_training=True):
    df = pd.read_csv(datasheet)
    classes = []
    if only_training:
        df = df[df.Split == "Train"]
    ys = df["y_true"].unique()
    class_map = {}
    for y in range(max(ys) + 1):
        class_map[y] = df["Species_Name"].loc[df["y_true"] == y].values[0]
        classes.append(class_map[y])
    rev_class_map = {value: key for key, value in class_map.items()}
    return class_map, rev_class_map


def sub_split(ids, ratio, iter_=None):
    test_size = int(len(ids) * ratio[2])
    if iter_ != None:
        multi = iter_ * test_size
        ids = np.roll(ids, multi)
    trainlist, vallist, testlist = np.split(
        ids, [int(ratio[0] * len(ids)), int((ratio[0] + ratio[1]) * len(ids))]
    )
    return trainlist, vallist, testlist


def split_generator(
    df,
    splitratio=[0.7, 0.15, 0.15],
    sort="Specimen_Id",
    target_col="y",
    unit_col="Specimen_Id",
    constraint="",
    split_col="Split",
    iter_n=None,
    verbose=False,
):
    """df (pandas)             should be a csv download coming from billy.
    sort (str)              a column by which to sort prior to assigning split
    split_col (str)         a column whose values dictate the split. eg if species is listed, the split ratio will be applied over each species
    splitratio ([float]*3)  train/val/test/split
    unit_col (str)          a column where unique values will be treated as a unit when assigning split
    """
    if sort in [None, ""]:
        df = df.sample(frac=1).reset_index(drop=True)
    elif sort == "same":
        pass
    else:
        df = df.sort_values(sort)
    df[split_col] = ""
    ys = df[target_col].unique()
    if -1 in ys:
        idx = np.where(ys == -1)
        ys = np.delete(ys, idx)
    if constraint != "":
        extras = df[constraint].unique()
    train = []
    val = []
    test = []

    for y in ys:
        sub_df = df[df[target_col] == y]
        if constraint != "":
            for extra in extras:
                subsub_df = sub_df[sub_df[constraint] == extra]
                units = subsub_df[unit_col].unique()
                trainlist, vallist, testlist = sub_split(units, splitratio, iter_n)
                train.extend(trainlist)
                val.extend(vallist)
                test.extend(testlist)
        else:
            units = sub_df[unit_col].unique()
            trainlist, vallist, testlist = sub_split(units, splitratio, iter_n)
            train.extend(trainlist)
            val.extend(vallist)
            test.extend(testlist)
    splits = {"Train": train, "Valid": val, "Test": test}
    for split in splits:
        for sid in splits[split]:
            matches = df[unit_col] == sid
            df.loc[matches, split_col] = split
    return df


# Owned by Johns Hopkins University, created prior to 5/28/2020


def alb_transform_test(imsize=256, p=1):
    albumentations_transform = Compose(
        [Resize(imsize, imsize), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
        p=1,
    )
    return albumentations_transform


def alb_transform_train(imsize=256, p=0.001, setting=0):
    if setting is None:
        setting = 0
    if setting == -1:
        albumentations_transform = Compose(
            [
                Resize(imsize, imsize),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            p=1,
        )
    if setting == 0:
        albumentations_transform = Compose(
            [
                Resize(imsize, imsize),
                RandomRotate90(),
                Flip(),
                Transpose(),
                GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                ),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            p=1,
        )
    if setting == 1:
        albumentations_transform = Compose(
            [
                Resize(imsize, imsize),
                RandomRotate90(),
                Flip(),
                Transpose(),
                GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                ),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            p=1,
        )
    elif setting == 2:
        albumentations_transform = Compose(
            [
                Resize(imsize, imsize),
                RandomRotate90(),
                Flip(),
                Transpose(),
                GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                ),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            p=1,
        )
    elif setting == 3:
        albumentations_transform = Compose(
            [
                Resize(imsize, imsize),
                RandomRotate90(),
                Flip(),
                Transpose(),
                GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                ),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            p=1,
        )
    elif setting == 4:
        albumentations_transform = Compose(
            [
                Resize(imsize, imsize),
                RandomRotate90(),
                Flip(),
                Transpose(),
                ColorJitter(
                    brightness=0.15, contrast=0, saturation=0.1, hue=0.025, p=0.7
                ),
                GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                ),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            p=1,
        )
    elif setting == 5:
        albumentations_transform = Compose(
            [
                Resize(imsize, imsize),
                RandomRotate90(),
                Flip(),
                Transpose(),
                ColorJitter(
                    brightness=0.15, contrast=0, saturation=0.1, hue=0.1, p=0.7
                ),
                GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                ),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            p=1,
        )
    elif setting == 30:
        albumentations_transform = Compose(
            [
                Resize(imsize, imsize),
                ColorJitter(
                    brightness=0.15, contrast=0, saturation=0.1, hue=0.1, p=0.7
                ),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            p=1,
        )
    return albumentations_transform
