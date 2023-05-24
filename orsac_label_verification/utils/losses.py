# Owned by Johns Hopkins University, created prior to 5/28/2020
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.losses import FocalLoss as FocalLossFastAI
from IPython.core.debugger import set_trace


class MosLoss(nn.Module):
    def __init__(self, loss_dict, num_classes):
        super(MosLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_dict = copy.deepcopy(loss_dict)
        assert len(self.loss_dict) > 0
        self.lweights = []
        self.mag_scale = []
        for k, v in self.loss_dict.items():
            assert "weight" in v
            assert "mag_scale" in v
            self.lweights.append(v["weight"])
            self.mag_scale.append(v["mag_scale"])
            del v["weight"]
            del v["mag_scale"]
        self.lweights = torch.tensor(self.lweights).cuda()
        self.mag_scale = torch.tensor(self.mag_scale).cuda()
        assert self.lweights.sum() == 1

        self.losses = [globals()[k](**v) for k, v in self.loss_dict.items()]

    def forward(self, inputs, targets, reduction=None):
        # if type(inputs) is list:
        #     inputs = inputs[0]
        if len(targets.shape) != len(inputs.shape):
            temp = torch.zeros(
                (targets.shape[0], self.num_classes), device=targets.device
            )
            temp.scatter_(1, targets.view(-1, 1), 1)
            targets = temp
        loss = self.losses[0](inputs, targets) * self.mag_scale[0]
        # print("l0:", loss)
        for i in range(1, len(self.losses)):
            l_i = self.losses[i](inputs, targets) * self.mag_scale[i]
            # print("l{}:".format(i), l_i)
            loss += l_i
        # print("Weights: ", self.lweights)
        loss = (loss * self.lweights).sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        print("Focal Loss with gamma = ", gamma)
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )

        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()
