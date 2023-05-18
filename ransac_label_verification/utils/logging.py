import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

from ransac_label_verification.utils.utils import split_generator

load_dotenv()

# This py file is for metric and results loggin, as well as path management.


def get_data_csv(data_csv):
    return os.path.join("data", data_csv)


def get_config(config):
    return os.path.join("configs", config)


def experiment_path(config):
    experiments_folder_directory = "experiments"
    if os.path.exists(experiments_folder_directory) == False:
        os.mkdir(experiments_folder_directory)
    exp_name = config.exp_name
    path = os.path.join(experiments_folder_directory, exp_name)
    return path


def get_img_abspath(img_path: str, config):
    if os.path.isabs(img_path):
        return img_path
    if config.base_directory == None:
        base_path = os.environ.get("IMAGE_BASE_DIRECTORY")
    else:
        base_path = config.base_directory
    return os.path.join(base_path, img_path)


def current_iter_path(config):
    return os.path.join(experiment_path(config), "Current_iter")


def sub_directory_path(config, sub_name):
    path = os.path.join(experiment_path(config), sub_name)
    return path


def setup_directories(config, sub_dir_list=None):
    if sub_dir_list == None:
        sub_dir_list = [
            "Data",
            "Preds",
            "Probs",
            "Current_iter",
            "Current_iter/Models",
            "Confusion_matrices",
        ]
    if not os.path.exists(experiment_path(config)):
        os.mkdir(experiment_path(config))
    for sub in sub_dir_list:
        path = sub_directory_path(config, sub)
        if not os.path.exists(path):
            os.mkdir(path)


def models_directory(config):
    return os.path.join(current_iter_path(config), "Models")


def model_weights_path(config):
    return os.path.join(models_directory(config), f"best_{config.best_weights}.pth")


def create_results_df(config):
    ransac_res_path = os.path.join(experiment_path(config), "results.csv")
    res_df = pd.read_csv(get_data_csv(config.data_csv_path))
    res_df["num_sampled"] = 0
    res_df["num_missed"] = 0
    for y in range(max(res_df.y.unique()) + 1):
        for j in range(config.top_x):
            res_df[f"{str(y)}_{j+1}"] = 0
    res_df.to_csv(ransac_res_path, index=False)


def init_data_df(config):
    data_path = os.path.join(current_iter_path(config), "data.csv")
    data_df = pd.read_csv(get_data_csv(config.data_csv_path))
    data_df.to_csv(data_path, index=False)


def save_current_iter(config, i):
    preds = pd.read_csv(os.path.join(current_iter_path(config), "preds.csv"))
    probs = pd.read_csv(os.path.join(current_iter_path(config), "probs.csv"))
    data = pd.read_csv(os.path.join(current_iter_path(config), "data.csv"))
    preds.to_csv(
        os.path.join(sub_directory_path(config, "Preds"), f"iter_{i}preds.csv"),
        index=False,
    )
    probs.to_csv(
        os.path.join(sub_directory_path(config, "Probs"), f"iter_{i}probs.csv"),
        index=False,
    )
    data.to_csv(
        os.path.join(sub_directory_path(config, "Data"), f"iter_{i}data.csv"),
        index=False,
    )


def init_metrics_df(config):
    path = os.path.join(experiment_path(config), "accuracies.csv")
    d = {"Test_Accuracies": [], f"Validation_{config.best_weights}": []}
    df = pd.DataFrame(data=d)
    df.to_csv(path, index=False)


def shuffle_data(config):
    df = pd.read_csv(os.path.join(current_iter_path(config), "data.csv"))
    new_df = split_generator(df, constraint="Species_Name", sort=None)
    new_df.to_csv(os.path.join(current_iter_path(config), "data.csv"), index=False)


def iter_data(config, i):
    df = pd.read_csv(os.path.join(current_iter_path(config), "data.csv"))
    new_df = split_generator(df, constraint="Species_Name", sort="same", iter_n=i)
    new_df.to_csv(os.path.join(current_iter_path(config), "data.csv"), index=False)


def save_test_output(config, preds, probs, acc):
    df = pd.read_csv(os.path.join(current_iter_path(config), "data.csv"))
    df = df[
        df.Id.apply(lambda x: os.path.exists(get_img_abspath(x, config)))
    ].reset_index(drop=True)
    df = df[df.Split == "Test"]
    df["pred"] = preds
    df.to_csv(os.path.join(current_iter_path(config), "preds.csv"), index=False)
    subm_df = df["Id"].copy()
    subm_df = pd.concat([subm_df, pd.DataFrame(data=probs)], axis=1)
    subm_df.to_csv(os.path.join(current_iter_path(config), "probs.csv"), index=False)
    save_metric(config, acc)


def save_metric(config, metric, metric_name="Test_Accuracies"):
    path = os.path.join(experiment_path(config), "accuracies.csv")
    metric_df = pd.read_csv(path)
    index = len([i for i in metric_df[metric_name] if str(i) != "nan"])
    metric_df.loc[index, metric_name] = metric
    metric_df.to_csv(path, index=False)


def log_results(config):
    preds = pd.read_csv(os.path.join(current_iter_path(config), "preds.csv"))
    prob_df = pd.read_csv(os.path.join(current_iter_path(config), "probs.csv"))
    preds["wrong"] = [
        preds.loc[i, "y"] != preds.loc[i, "pred"] for i in range(len(preds))
    ]
    path = os.path.join(experiment_path(config), "results.csv")
    df = pd.read_csv(path)

    for temp_id in preds["Id"]:
        df.loc[df["Id"] == temp_id, "num_missed"] += preds.loc[
            preds["Id"] == temp_id, "wrong"
        ].item()

        df.loc[df["Id"] == temp_id, "num_sampled"] += 1
        probs = (
            prob_df.loc[
                prob_df.Id == temp_id,
                [str(i) for i in range(max(df["y"].unique()) + 1)],
            ]
            .values.flatten()
            .tolist()
        )
        probs = np.array(probs)
        top_k = -1 * config.top_x
        top = np.argsort(probs)[top_k:]
        top = np.flip(top)
        for j, i in enumerate(top):
            df.loc[df.Id == temp_id, f"{str(i)}_{j+1}"] += 1
    df.to_csv(path, index=False)


def record_top_x_pred(config, probs_path=None):
    top_x = config.top_x
    if probs_path == None:
        df = pd.read_csv(os.path.join(current_iter_path(config), "probs.csv"))
    else:
        df = pd.read_csv(probs_path)
    top_preds_df = pd.read_csv(os.path.join(experiment_path(config), "top_preds.csv"))
    results = pd.read_csv(os.path.join(experiment_path(config), "results.csv"))
    for image in df.Id.unique():
        probs = (
            df.loc[
                df.Id == image, [str(i) for i in range(max(results["y"].unique()) + 1)]
            ]
            .values.flatten()
            .tolist()
        )
        row = top_preds_df.loc[top_preds_df.Id == image]
        probs = np.array(probs)

        top_k = -1 * top_x
        top = np.argsort(probs)[top_k:]
        top = np.flip(top)

        for j, i in enumerate(top):
            top_preds_df.loc[top_preds_df.Id == image, f"{str(i)}_{j+1}"] += 1

    top_preds_df.to_csv(
        os.path.join(experiment_path(config), "top_preds.csv"), index=False
    )
    return top_preds_df


def create_top_preds_df(config):
    top_preds_df = pd.read_csv(os.path.join(experiment_path(config), "results.csv"))
    classes = [i for i in top_preds_df.y.unique() if i != -1]
    for i in classes:
        for j in range(config.top_x):
            top_preds_df[f"{str(i)}_{j+1}"] = 0
    top_preds_df.to_csv(
        os.path.join(experiment_path(config), "top_preds.csv"), index=False
    )


# Generates and saves confusion matrix of iteration iter's test results to experiment folder.
def con_mat_iter(config, y_true, y_pred, iter):
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=list(config.class_names)
    ).plot()
    plt.xticks(rotation=90)
    plt.gcf().set_size_inches(25, 18)
    plt.savefig(
        os.path.join(
            experiment_path(config), "Confusion_matrices", f"con_mat_iter{iter}"
        )
    )
