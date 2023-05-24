import torch
from sklearn.metrics import accuracy_score, f1_score


def accuracy(preds, targets):
    return torch.tensor(
        accuracy_score(preds.cpu().numpy().argmax(1), targets.cpu().numpy())
    )


def macro_f1(preds, targets):
    return torch.tensor(
        f1_score(preds.cpu().numpy().argmax(1), targets.cpu().numpy(), average="macro")
    )
