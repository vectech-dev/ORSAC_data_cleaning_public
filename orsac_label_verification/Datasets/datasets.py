from __future__ import division, print_function
import time
import math
import os
import traceback

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import transform
from torch.utils.data import Dataset
from tqdm import tqdm

from orsac_label_verification.utils.logging import get_img_abspath
from orsac_label_verification.Datasets.preprocessing import *




def load_image(impath, pil=False):
    if pil:
        image = Image.open(impath)
    else:
        image = cv2.imread(impath, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found at {}".format(impath))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class MosDataset(Dataset):
    def __init__(self, config, data_df, transformer=None, one_hot_label=False):
        """
        Params:
            data_df: data DataFrame of image name and labels
            imsize: output image size
        """
        super().__init__()
        self.tfms = None
        self.config = config
        self.num_classes = self.config.num_classes
        self.transformer = transformer

        if config.sampling == "oversampling":
            train_df = data_df[data_df["Split"] == "Train"].reset_index(drop=True)
            valid_df = data_df[data_df["Split"] == "Valid"].reset_index(drop=True)
            test_df = data_df[data_df["Split"] == "Test"].reset_index(drop=True)
            if not train_df.empty:
                oversampled_train_df = self.oversample(train_df)

                oversampled_df = pd.concat(
                    [oversampled_train_df, valid_df, test_df]
                ).reset_index(drop=True)

                # oversampled_df.to_csv('tmp/mid.csv')
                df_summary = pd.DataFrame()
                for split in data_df["Split"].unique():
                    if split is not None:
                        temp = data_df.loc[data_df["Split"] == split]
                        df_summary[split] = temp["y"].value_counts()
                        if split == "Train":
                            df_summary["Oversampled Train"] = oversampled_df[
                                "y"
                            ].value_counts()
                df_summary.fillna(0, inplace=True)
                summary_dtype = {x: "int" for x in df_summary.columns}
                df_summary = df_summary.astype(summary_dtype)
                df_summary = df_summary.sort_index(ascending=True)
                print(df_summary)

                self.images_df = oversampled_df
            else:
                self.images_df = data_df
        else:
            self.images_df = data_df

        self.one_hot_label = one_hot_label
        self.images_df["Id"] = self.images_df.Id.apply(
            lambda x: get_img_abspath(x, config)
        )
        # see how many don't exist
        counts = self.images_df["Split"].value_counts()
        print("Images in datasheet:\n{}".format(counts))
        self.images_df = self.images_df[
            self.images_df.Id.apply(os.path.exists)
        ].reset_index(drop=True)

        counts = self.images_df["Split"].value_counts()
        print("Existing images:\n{}".format(counts))
        self.transforms=[CV2Resize((self.config.imsize,self.config.imsize))]
        if self.config.white_balance:
            self.transforms.insert(0,WhiteBalance())
        if self.config.preload_data:
            print("Preloading images...")
            self.imarray = np.zeros(
                (len(self.images_df), self.config.imsize, self.config.imsize, 3),
                dtype="uint8",
            )
            for idx, impath in enumerate(tqdm(self.images_df["Id"])):
                img = load_image(impath)
                img = T.Compose(self.transforms)(image)
                image = np.array(image)
       
                # cv2.imwrite(f"image_{time.time}.png",image)
                self.imarray[idx, :, :, :] = img

    def __len__(self):
        return len(self.images_df)

    def get_labels(self):
        return self.images_df["y"]

    def __getitem__(self, idx):
        imagename = self.images_df.loc[idx, "Id"]
       
        if self.config.preload_data:
            image = self.imarray[idx, :, :, :]
        else:
            image = load_image(imagename)
            image = T.Compose(self.transforms)(image)
            image = np.array(image)
            print(os.getcwd())
            cv2.imwrite(f"image_{time.time()}.jpg",image)
            
        label = self.images_df["y"][idx]

        if self.transformer:
            image = self.transformer(image=image)["image"]
        else:
            image = transform.resize(image, (self.config.imsize, self.config.imsize))
        # print(image.shape)
        image = torch.from_numpy(image).float()
        image = image.permute(-1, 0, 1)
        return image, label

    def getimage(self, idx):
        image, targets = self.__getitem__(idx)
        image = image.permute(1, 2, 0).numpy()
        imagename = self.images_df.loc[idx, "Id"]
        return image, targets, imagename

    # oversampling function
    def oversample(self, df: pd.DataFrame):
        # define number of species
        num_of_spec = self.config.num_classes
        non_random_df = pd.DataFrame()
        oversampled_df = pd.DataFrame()
        remainder = [None] * num_of_spec

        # loop through each species
        for i in range(0, num_of_spec):
            # create temp dataframe
            temp_df = pd.DataFrame()

            # selecting rows based on species (0 = Rh. Sang., 1 = Amb. Varieg. 2 = Derm. Variab., 3 = Rh. Pulch. 4 = Derm. Nit.)
            temp_df = df[df["y"] == i]

            # max train samples
            max_train_samples = df["y"].value_counts().max()

            # count current train samples
            # current_train_samples = temp_df['Split'].value_counts()['Train']
            try:
                current_train_samples = temp_df["Split"].value_counts()["Train"]
                print("success for :", i)
            except Exception as e:
                print("oversampling exited with error on species ", i)
                print(e)
                print(traceback.format_exc())
                continue

            # use non-random sampling up to the whole number of the ratio between max train sample and current train sample
            non_random_df = pd.DataFrame(
                np.repeat(
                    temp_df.values, max_train_samples / current_train_samples, axis=0
                )
            )
            non_random_df.columns = temp_df.columns

            # keep track of remainder of ratios
            remainder[i] = abs(
                math.trunc(max_train_samples / current_train_samples)
                - max_train_samples / current_train_samples
            )

            # append non-random df to oversampled df
            oversampled_df = pd.concat([oversampled_df, non_random_df]).reset_index(
                drop=True
            )

        # random sampling
        lst = [oversampled_df]
        # go through each data sample within a species
        for class_index, group in df.groupby("y"):
            # sample a fraction of each of the species by using the remainder from non-random sampling and add it to the non-random oversampled df
            lst.append(group.sample(frac=remainder[class_index], replace=True))

        # shuffle the oversampled df
        final_df = pd.concat(lst).reset_index(drop=True)
        shuffled_df = final_df.sample(frac=1).reset_index(drop=True)

        # return final_df
        return shuffled_df
