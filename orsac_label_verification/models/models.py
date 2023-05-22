import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from pretrainedmodels import xception


class xception_mod(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet"):
        super().__init__()
        self.net = xception(pretrained=pretrained)
        self.net.last_linear = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True
        )

    def forward(self, x):
        return self.net(x)


class efficientnet_mod(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        model = EfficientNet.from_pretrained("efficientnet-b0")
        self.net = model

        self.net._fc = nn.Linear(
            in_features=self.net._fc.in_features, out_features=num_classes, bias=True
        )

    def forward(self, x):
        return self.net(x)
