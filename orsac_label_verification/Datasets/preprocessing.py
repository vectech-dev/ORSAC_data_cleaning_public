import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image



class CV2Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        # Convert PIL Image to numpy array (H x W x C)
        img_np = np.array(img)
        # Resize using cv2
        img_resized = cv2.resize(img_np, self.size, interpolation=cv2.INTER_AREA)       
        # Convert numpy array back to PIL Image
        return Image.fromarray(img_resized)




class SquaredImage:
    def __init__(self,val:int=-1):
        self.val=val

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.make_square(img,self.val)

    @staticmethod
    #EDITTED
    def make_square(img,val):
        img = np.array(img)
        rolled=False
        if img.shape[0] > img.shape[1]:
            img = np.rollaxis(img, 1, 0)
            rolled=True
        toppadlen = (img.shape[1] - img.shape[0]) // 2
        bottompadlen = img.shape[1] - img.shape[0] - toppadlen
        if val == -1:
            toppad = img[:5, :, :].mean(0, keepdims=True).astype(img.dtype)
            toppad = np.repeat(toppad, toppadlen, 0)
            bottompad = img[-5:, :, :].mean(0, keepdims=True).astype(img.dtype)
            bottompad = np.repeat(bottompad, bottompadlen, 0)
        else:
            toppad=val*np.ones((toppadlen,img.shape[1],3),dtype=np.uint8)
            bottompad=val*np.ones((bottompadlen,img.shape[1],3),dtype=np.uint8)
            
        img = np.concatenate((toppad, img, bottompad), axis=0)
        if rolled:
            img = np.rollaxis(img, 1, 0)
        return Image.fromarray(np.uint8(img))


class WhiteBalance:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.balance_tray(img, perc=self.perc)

    def __init__(self, perc=0.02) -> None:
        self.perc = perc

    def moving_average_other(self, a, n=3):
        dif = int((n - 1) / 2)
        for i in range(dif):
            a = np.append(a, a[-1])
            a = np.append(a[0], a)
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[dif * 2 :] / n

    def compress_pad(self, pad, get_len=False):
        """receive a pad of any size and return it as a vector"""
        pad_shape = pad.shape
        if get_len:
            if pad_shape[0] > pad_shape[1]:
                res = np.mean(pad, axis=1)
                return res, len(res)
            elif pad_shape[1] > pad_shape[0]:
                return np.mean(pad, axis=0)
        else:
            if pad_shape[0] > pad_shape[1]:
                return np.mean(pad, axis=1)
            elif pad_shape[1] > pad_shape[0]:
                return np.mean(pad, axis=0)

    def each_channel_mv_avg(self, arr, n_avg=51):
        for j, ch in enumerate(["B", "G", "R"]):
            arr[:, j] = self.moving_average_other(arr[:, j], n=n_avg)
        return arr

    def balance_tray(
        self,
        img,
        outdir="./",
        method="fancy",
        pad_size=3,
        perc=0.02,
        flat_target=215,
    ):
        """Balance the tray of a image"""

        full_img = img = np.array(img)

        if outdir is not None and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        if type(full_img) != type(None):
            ogshape = full_img.shape
            img = cv2.resize(
                full_img, dsize=(int(ogshape[1] * perc), int(ogshape[0] * perc))
            )
            shape = img.shape
            ct_avg = self.each_channel_mv_avg(self.compress_pad(img[0:pad_size, :, :]))
            cb_avg = self.each_channel_mv_avg(self.compress_pad(img[pad_size - shape[1] : shape[1], :, :]))
            cl_avg = self.each_channel_mv_avg(self.compress_pad(img[:, 0:pad_size, :]))
            cr_avg = self.each_channel_mv_avg(self.compress_pad(img[:, pad_size - shape[0] : shape[0], :]))
            if method == "simple":
                simple_mult = (
                    (
                        np.mean(ct_avg, axis=0)
                        + np.mean(cb_avg, axis=0)
                        + np.mean(cl_avg, axis=0)
                        + np.mean(cr_avg, axis=0)
                    )
                    / 4
                    / flat_target
                )
                res = full_img / simple_mult
            elif method == "fancy":
                tweight = np.arange(len(cr_avg) - 1, -1, -1)
                bweight = np.arange(0, len(cr_avg), 1)
                lweight = np.arange(len(ct_avg) - 1, -1, -1)
                rweight = np.arange(0, len(ct_avg), 1)
                shape = img.shape
                weight_img = np.zeros(shape)
                for r in range(shape[0]):
                    for c in range(shape[1]):
                        # for ch in range(shape[2]):
                        weight_img[r, c, :] = (
                            ct_avg[c, :] * tweight[r]
                            + cr_avg[r, :] * rweight[c]
                            + cb_avg[c, :] * bweight[r]
                            + cl_avg[r, :] * lweight[c]
                        ) / (tweight[r] + bweight[r] + lweight[c] + rweight[c])
                mult_img = weight_img / (flat_target)
                mult_img = cv2.resize(mult_img, (ogshape[1], ogshape[0]))
                res = full_img / mult_img

            res = np.array(res)
            res = np.clip(res, a_min=0, a_max=255)
            return Image.fromarray(np.uint8(res))