import json
import os

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from dotenv import load_dotenv

from ransac_label_verification.Datasets.loaders import get_test_loader
from ransac_label_verification.models.loaders import load_model
from ransac_label_verification.train_config import ExperimentationConfig
from ransac_label_verification.utils.logging import get_config


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))


load_dotenv()


def test(config, val_mode=False):
    Model = config.get_model()
    model = Model(**config.model_kwargs)
    trained_model = load_model(config, model)

    if not val_mode:
        test_loader = get_test_loader(config)
    else:
        test_loader = get_test_loader(config, split="Valid")
    prob_i = 0
    running_corrects = 0
    all_probs = torch.Tensor(len(test_loader.dataset), config.num_classes)
    all_preds = torch.Tensor(len(test_loader.dataset))
    all_labels = torch.Tensor(len(test_loader.dataset))
    for inputs, labels in test_loader:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        with torch.set_grad_enabled(False):
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)

            all_probs[prob_i : prob_i + outputs.shape[0], :] = outputs
            all_preds[prob_i : prob_i + outputs.shape[0]] = preds
            all_labels[prob_i : prob_i + outputs.shape[0]] = labels
            prob_i += outputs.shape[0]
            running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects / len(test_loader.dataset)

    print("Test Accuracy", acc)
    return all_probs, all_preds, all_labels, acc.item()


if __name__ == "__main__":
    path = "example_config.json"
    with open(get_config(path)) as json_file:
        settings_json = json.load(json_file)
    print(settings_json)
    config = ExperimentationConfig.parse_obj(settings_json)
    test(config)
