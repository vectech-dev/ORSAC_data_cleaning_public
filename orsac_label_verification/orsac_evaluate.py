import copy
import json
import os
from argparse import ArgumentParser

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from orsac_label_verification.orsac_test import OrsacResults
from orsac_label_verification.test import test
from orsac_label_verification.train import train_eval
from orsac_label_verification.train_config import ExperimentationConfig
from orsac_label_verification.utils.logging import (
    con_mat_iter,
    experiment_path,
    get_data_csv,
    init_data_df,
    init_metrics_df,
    setup_directories,
)
from orsac_label_verification.utils.utils import get_classes, split_generator


def save_cleaned_set(config):
    exp_path = experiment_path(config)
    data_df = pd.read_csv(get_data_csv(config.data_csv_path))
    unclean_path = f"unclean_{config.data_csv_path}"
    clean_path = f"clean_{config.data_csv_path}"
    fig_path = os.path.join(exp_path, "figures")

    if os.path.exists(exp_path):
        if os.path.exists(fig_path):
            flagged_df = pd.read_csv(os.path.join(fig_path, "flagged_specimens.csv"))
        else:
            results = OrsacResults(config, 1, True)
            flagged_df = results.save_results_df(verbose=True)
        cleaned_df = data_df.set_index("Specimen_Id")
        cleaned_df = cleaned_df.drop(flagged_df.Specimen_Id.unique())
        cleaned_df["Specimen_Id"] = cleaned_df.index
        cleaned_df.set_index(
            pd.Index([i for i in range(len(cleaned_df))]), inplace=True
        )
        # Here, put both cleaned and data_df through the split generator with test set size equal to zero and shuffle set to true.
        data_df = split_generator(data_df, splitratio=[0.85, 0.15, 0], sort=None)
        cleaned_df = split_generator(cleaned_df, splitratio=[0.85, 0.15, 0], sort=None)
        cleaned_df.to_csv(get_data_csv(clean_path), index=False)
        data_df.to_csv(get_data_csv(unclean_path), index=False)
        return clean_path, unclean_path
    else:
        print("Experiment does not exist")


def save_eval_results(config, c_acc, unc_acc):
    path = os.path.join(experiment_path(config), "eval_accuracies.csv")
    d = {"Clean Accuracy": [], f"Unclean Accuracy": []}
    df = pd.DataFrame(data=d)
    df.loc[0, "Clean Accuracy"] = c_acc
    df.loc[0, "Unclean Accuracy"] = unc_acc
    df.to_csv(path, index=False)


def init_eval(config):
    setup_directories(config)
    init_data_df(config)
    init_metrics_df(config)
    with open(os.path.join(experiment_path(config), "config.json"), "w") as outfile:
        json.dump(config.dict(exclude_none=True), outfile)

def _evaluation(settings_json):
    class_dict, _ = get_classes(get_data_csv(settings_json["data_csv_path"]))
    settings_json["num_classes"] = len(class_dict)
    settings_json["class_names"] = list(class_dict.values())
    settings_json["model_kwargs"]["num_classes"] = len(class_dict)
    clean_json = copy.deepcopy(settings_json)
    clean_json["exp_name"] = os.path.join(clean_json["exp_name"], "clean_exp")
    unclean_json = copy.deepcopy(settings_json)
    unclean_json["exp_name"] = os.path.join(unclean_json["exp_name"], "unclean_exp")
    print(settings_json)
    config = ExperimentationConfig.parse_obj(settings_json)
    clean_path, unclean_path = save_cleaned_set(config)
    unclean_json["data_csv_path"] = unclean_path
    clean_json["data_csv_path"] = clean_path
    clean_config = ExperimentationConfig.parse_obj(clean_json)
    unclean_config = ExperimentationConfig.parse_obj(unclean_json)
    unclean_acc = evaluate(unclean_config)
    print(f"Uncleaned Data Model Test Accuracy: {unclean_acc}")
    clean_acc = evaluate(clean_config)
    print(f"Cleaned Data Model Test Accuracy: {clean_acc}")
    save_eval_results(config, clean_acc, unclean_acc)


def evaluate(config):
    init_eval(config)
    train_eval(config)
    _, preds, labels, acc = test(config,mode='Eval')
    con_mat_iter(config, labels, preds, 0)
    return acc


def orsac_evaluate(experiment_dir: str):
    with open(experiment_dir + "/config.json", "r") as json_file:
        settings_json = json.load(json_file)
        settings_json["mode"] = "eval"  # Change the mode to "test"
        _evaluation(settings_json)
