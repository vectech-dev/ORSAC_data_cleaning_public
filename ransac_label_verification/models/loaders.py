import torch

from ransac_label_verification.utils.logging import model_weights_path


def load_model(config, model, test=True):
    if test:
        # load state dict
        state_dict_obj = torch.load(
            model_weights_path(config), map_location=config.device
        )
        # net = load_model_weights(net, state_dict_obj)
        try:
            model.load_state_dict(state_dict_obj, strict=False)
        except:
            model.load_state_dict(state_dict_obj["model"], strict=False)
    model.to(config.device)
    model.eval()
    return model
