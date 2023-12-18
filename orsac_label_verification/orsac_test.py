import argparse
import json
import os
import random as rand
import threading
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages
from natsort import natsorted
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from orsac_label_verification.train_config import ExperimentationConfig
from orsac_label_verification.utils.logging import (
    current_iter_path,
    experiment_path,
    get_config,
    get_data_csv,
    record_top_x_pred,
    sub_directory_path,
)
from orsac_label_verification.utils.utils import get_classes, str_to_list

load_dotenv()





class OrsacResults:
    def __init__(self, config, threshold, test_mode=True):
        self.config = config
        self.directory = experiment_path(config)
        self.threshold = threshold
        self.test_mode = test_mode
        res_path = os.path.join(experiment_path(config), "results.csv")
        self.results = pd.read_csv(res_path)
        # maybe change the get_classes method to be uniform throughout code database. Ie, only define it once and reuse it.
        self.class_map, self.rev_class_map = get_classes(res_path)
        self.figure_path = os.path.join(self.directory, "figures")
        sampled = self.results[self.results["num_sampled"] > 0].copy()

        num_sampled = np.array(sampled["num_sampled"].values.tolist())
        num_missed = np.array(sampled["num_missed"].values.tolist())
        freq = num_missed / num_sampled
        sampled.loc[:, "freq"] = freq
        sampled.loc[sampled["freq"] >= self.threshold, "mislab_pred"] = True
        sampled.loc[sampled["freq"] < self.threshold, "mislab_pred"] = False
        for spec_id in sampled.loc[
            sampled["mislab_pred"] == True, "Specimen_Id"
        ].unique():
            sampled.loc[sampled["Specimen_Id"] == spec_id, "save"] = True

        self.sampled = sampled
        # self.acc_fig, accs = self.graph_accuracies()
        self.title_slide = self.title_fig()

        if os.path.exists(self.figure_path) != True:
            os.makedirs(self.figure_path)

    def save_figures(self):
        with PdfPages(os.path.join(self.figure_path, "All_figures.pdf")) as fig_pdf:
            fig_pdf.savefig(self.title_slide)
            if self.test_mode:
                fig_pdf.savefig(self.graph_distribution())
                fig_pdf.savefig(
                    self.save_confusion_matrix(norm=None), bbox_inches="tight"
                )
                fig_pdf.savefig(
                    self.save_confusion_matrix(norm="true"), bbox_inches="tight"
                )
                fig_pdf.savefig(
                    self.save_confusion_matrix(norm="pred"), bbox_inches="tight"
                )
                fig_pdf.savefig(
                    self.save_confusion_matrix(norm=None), bbox_inches="tight"
                )
                fig_pdf.savefig(
                    self.save_confusion_matrix(norm="true"), bbox_inches="tight"
                )
                fig_pdf.savefig(
                    self.save_confusion_matrix(norm="pred"), bbox_inches="tight"
                )

                caught_percents_fig, caught_percents_df = self.get_caught_percents()
                # fig_pdf.savefig(self.acc_fig, bbox_inches="tight")
                fig_pdf.savefig(caught_percents_fig, bbox_inches="tight")
                fig_pdf.savefig(self.graph_positives_by_species(), bbox_inches="tight")

            else:
                fig_pdf.savefig(self.graph_distribution())
                fig_pdf.savefig(self.graph_positives_by_species(), bbox_inches="tight")

    def save_results_df(self, verbose=False):
        flagged = self.sampled[self.sampled["save"] == True].copy()
        replace_dict = {str(i): self.class_map[i] for i in self.class_map}
        new_flagged = flagged.rename(replace_dict, axis="columns")
        # where this gets saved to needs to change
        new_flagged.to_csv(
            os.path.join(self.figure_path, "flagged_specimens.csv"),
            index=False,
        )
        if verbose:
            return new_flagged

    def title_fig(self):
        fig = plt.figure()
        fig.text(0.05, 0.9, str(self.config.exp_name))
        fig.text(0.05, 0.85, f"Epochs:{str(self.config.epochs)}")
        fig.text(0.05, 0.8, f"Dataset:{str(self.config.data_csv_path)}")
        flagged_frame = self.sampled[self.sampled["mislab_pred"] == True]
        num_flagged = len(flagged_frame["Specimen_Id"].unique())
        num_tested = len(self.sampled["Specimen_Id"].unique())
        fig.text(0.05, 0.75, f"Iterations:{str(self.config.n_iterations)}")
        fig.text(0.05, 0.7, f"Flagging Threshold:{str(self.threshold*100)}%")
        fig.text(0.05, 0.65, f"Specimens Flagged:{str(num_flagged)}")
        fig.text(0.05, 0.6, f"Specimens Tested:{str(num_tested)}")
        plt.close("all")
        return fig
#Currently this is unused, so check back about whether you need it or not. Gut says no, not a useful figure to generate. 
#It also eats up  a lot of memory. Look for ways to reduce memory usage. 
    def top_x_all(self):
        self.top_pred_df = self.results.copy()
        classes = [i for i in self.results.y.unique() if i != -1]
        for i in classes:
            self.top_pred_df[str(i)] = [
                [0] * self.config.top_x for i in range(len(self.top_pred_df))
            ]
        # Here is where the mechanics have changed. If problems when running check here.

        for path in os.list_dir(sub_directory_path(self.config, "Probs")):
            record_top_x_pred(self.config, probs_path=path)

        self.top_pred_df.to_csv(
            os.path.join(experiment_path(self.config), "top_preds.csv"), index=False
        )

    def str_to_list(self, string):
        string = string.replace("[", "")
        string = string.replace("]", "")
        string = string.split(",")
        string = [int(i) for i in string]
        return string

    # needs editing to work for non front/back view.
    def save_top_x_figs(self):
        with PdfPages(
            os.path.join(self.figure_path, f"All top{self.config.top_x} preds.pdf")
        ) as samples_pdf:
            samples_pdf.savefig(self.title_slide)
            for y in tqdm(range(max(self.results["y"].unique()) + 1)):
                sub_df = self.sampled[self.sampled.y == y]
                for sample in sub_df.loc[
                    sub_df["mislab_pred"] == True, "Specimen_Id"
                ].unique():
                    for view in sub_df.View.unique():
                        if (
                            view
                            not in sub_df[sub_df.Specimen_Id == sample].View.unique()
                        ):
                            continue
                        pred_dist = self.get_top_x_pred_fig(sample, view)
                        if pred_dist == None:
                            continue
                        samples_pdf.savefig(pred_dist, bbox_inches="tight")
#consider removing top_x_pred functionality for this repo, as it is clunky and unnecessary. 
    def get_top_x_pred_fig(self, sample, view):
        # top_pred_df=pd.read_csv(os.path.join(experiment_path(self.config),'results.csv'))
        top_pred_df = self.sampled
        sample_rows = top_pred_df.loc[top_pred_df["Specimen_Id"] == sample]
        row = sample_rows[sample_rows.View == view]

        x_axis = []
        xs = np.arange(max(self.results["y"].unique()) + 1)
        xs = xs * 1.7
        xs_bar = xs - 0.5
        y_axes = [[] for i in range(self.config.top_x)]

        if self.test_mode:
            spec_true = self.class_map[sample_rows["y_true"].unique()[0]]
        spec_lab = self.class_map[sample_rows["y"].unique()[0]]

        for i in range(max(self.results["y"].unique()) + 1):
            x_axis.append(self.class_map[i])

            for j, ax in enumerate(y_axes):
                ax.append(row[f"{str(i)}_{j+1}"].values[0])

        fig = plt.figure(figsize=(20, 5))
        plt.title(f"Specimen {sample}{view} Prediction Counts")
        plt.xlabel("Species")
        plt.ylabel("Prediction Count")
        plt.figtext(0.2, 1.0, f"labeled species : {spec_lab}")
        if self.test_mode:
            plt.figtext(0.2, 0.9, f"True Species :{spec_true}")

        if row["mislab_pred"].values[0] == True:
            plt.figtext(0.2, 0.8, "Flagged View")
        bar_list = []
        width = 0.30
        hlist = ["", "", "", "", ""]
        clist = ["red", "blue", "black", "gray", "white"]
        for i, y_axis in enumerate(y_axes):
            bar_list.append(
                plt.bar(
                    xs_bar + width * i,
                    y_axis,
                    width,
                    align="center",
                    color=clist[i],
                    hatch=hlist[i],
                    edgecolor="black",
                    fill=True,
                    label=f"Pred #{i+1} Count",
                )
            )
            # bar_list.append(plt.barh(xs_bar+width*i,y_axis,width,align='edge',color=clist[i],hatch=hlist[i],edgecolor='black',fill=True,label=f'Pred #{i+1} Count'))
        for j, i in enumerate(xs_bar):
            # plt.axhline(i+width*5+.05,color='black',linestyle='dashed')
            if j == 0:
                plt.axvline(i - 0.15, color="black", linestyle="dashed")
            plt.axvline(i + width * 5 - 0.01, color="black", linestyle="dashed")

        plt.xticks(xs, x_axis, rotation="vertical")
        plt.legend(fontsize=17)
        plt.close(fig)
        return fig
#The below functionis unused REMOVE
    # def record_top_x_pred(self, path, top_x=3):
    #     df = pd.read_csv(path)

    #     for image in df.Id.unique():
    #         probs = (
    #             df.loc[
    #                 df.Id == image,
    #                 [str(i) for i in range(max(self.results["y"].unique()) + 1)],
    #             ]
    #             .values.flatten()
    #             .tolist()
    #         )
    #         probs = np.array(probs)

    #         top_k = -1 * top_x
    #         top = np.argsort(probs)[top_k:]
    #         top = np.flip(top)

    #         image = image.replace("/opt/ImageBase/mosID-production/", "")

    #         for i, j in enumerate(top):
    #             self.top_pred_df.loc[self.top_pred_df.Id == image, str(j)].values[0][
    #                 i
    #             ] += 1
    #             row = self.top_pred_df.loc[self.top_pred_df.Id == image, str(j)].values[
    #                 0
    #             ]
    #     return None
# Below is code dealing with graphing accuracies over all iterations.
#  Worthwhile making a new one that tracks val too

    # def get_iter_acc(self, ipath):
    #     with open(ipath) as f:
    #         lines = f.readlines()
    #         return float(lines[0].split(" ")[2])

    # # This could be kept, the only thing is you have to change where it looks for accuracies.
    # def graph_accuracies(self):
    #     accs = []
    #     for sub in [i for i in natsorted(os.listdir(self.directory)) if "iter" in i]:
    #         path = os.path.join(self.directory, sub, "preds/results.txt")
    #         if os.path.exists(path):
    #             accs.append(self.get_iter_acc(path))
    #     x = [i for i in range(len(accs))]
    #     y = accs

    #     plt.title("Model Accuracy by Iteration")
    #     fig = plt.figure()
    #     plt.scatter(x, y)
    #     plt.xlabel("Iteration Number")
    #     plt.ylabel("Percent accuracy")

    #     plt.close("all")
    #     return fig, accs


    def get_caught_percents(self):
        mislabeld = self.sampled[self.sampled["mislabeled"] == True]
        p_df = self.sampled[self.sampled["mislab_pred"] == True]
        tp_df = p_df[p_df["mislabeled"] == True]
        percents = []
        for species in mislabeld["Species_Name"].unique():
            total = mislabeld["Species_Name"].value_counts()[species]
            if species not in tp_df["Species_Name"].unique():
                part = 0
            else:
                part = tp_df["Species_Name"].value_counts()[species]
            percents.append([species, part, total, part / total])
        percents_df = pd.DataFrame(
            percents, columns=["Species_Name", "Flagged", "Mislabeled", "Percent"]
        )
        fig = plt.figure()
        ax1 = plt.axes()
        ax = percents_df.plot(
            x="Species_Name", y=["Flagged", "Mislabeled"], kind="bar", ax=ax1
        )
        fig.add_axes(ax)

        plt.title("Flagged and Mislabeled Counts by Species")
        plt.close("all")

        return fig, percents_df

    # should work the same way.
    def graph_positives_by_species(self):
        p_df = self.sampled[self.sampled["mislab_pred"] == True]
        if self.test_mode:
            fp_df = p_df[p_df["mislabeled"] == False]
            y_spec = fp_df["Species_Name"]
            y = []
            part_label = "False Positives"
        else:
            y_spec = p_df["Species_Name"]
            y = []
            part_label = "Positives"

        z_spec = self.sampled["Species_Name"]

        z = []
        for species in z_spec.unique():
            z.append(z_spec.value_counts()[species])

            if species not in y_spec.unique():
                y.append(0)
            else:
                y.append(y_spec.value_counts()[species])

        X_axis = np.arange(len(z_spec.unique()))

        fig = plt.figure(figsize=(20, 5))
        plt.title(f"{part_label} counts by Species")

        plt.bar(X_axis - 0.2, y, 0.4, label=part_label)
        plt.bar(X_axis + 0.2, z, 0.4, label="Total")

        plt.xticks(X_axis, list(z_spec.unique()), rotation="vertical")
        plt.legend()

        plt.close("all")
        return fig

    # should work the same way.
    def save_confusion_matrix(self, norm=None):
        matrix = confusion_matrix(
            self.sampled["mislabeled"].values.tolist(),
            self.sampled["mislab_pred"].values.tolist(),
            normalize=norm,
        )

        df_cm = pd.DataFrame(
            matrix, ["Correctly Labeled", "Mislabeled"], ["Unflagged", "Flagged"]
        )

        title = f"Confusion Matrix {norm} Normalized(Image Level)"
        sn.set(font_scale=1.4)
        fig = plt.figure()
        ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        fig.add_axes(ax)
        plt.title(title)
        plt.close("all")

        return fig

    # should work the same way.
    def graph_distribution(self):
        bin_num = min(self.sampled["num_sampled"].unique())
        fig = plt.figure()
        ax1 = plt.axes()
        if self.test_mode:
            ax = self.sampled.hist(
                column="freq", by=["mislabeled"], bins=bin_num, ax=ax1
            )
            ax[0].set_title("Unaltered Samples")
            ax[1].set_title("Mislabeled Samples")
            fig.add_axes(ax[0])
            fig.add_axes(ax[1])
        else:
            ax = self.sampled.hist(column="freq", bins=bin_num, ax=ax1)
            fig.add_axes(ax1)
        plt.close("all")
        return fig



# This works well for small percentages.
def iterate_test_sheet(path, percent_missed, i):
    # generate test sheet by random mislabeling some percentage of the data in config datasheet.
    # to simplify things, make dummy column of y and call true column of  y called y_true, then
    # make mislabeld column.
    df = pd.read_csv(path)
    df["y_true"] = df["y"]
    df["mislabeled"] = False
    ys = df["y_true"].unique()

    for clas in [x for x in ys if x != -1]:
        units = df.loc[df.y == clas, "Id"].unique()
        misclass = units[
            int(len(units) * percent_missed * i) : int(
                len(units) * percent_missed * (i + 1)
            )
        ]
        for specimen in misclass:
            df.loc[df.Id == specimen, "y"] = rand.choice(
                [i for i in range(len(ys)) if i != int(clas)]
            )
            df.loc[df.Id == specimen, "mislabeled"] = True
    return df


def generate_test_sheet(df, percent_missed):
    df["y_true"] = df["y"]
    df["mislabeled"] = False
    ys = df["y_true"].unique()
    ys = [i for i in ys if i != -1]
    for clas in [x for x in ys if x != -1]:
        misclass = rand.sample(
            sorted(df.loc[df["y"] == clas].index),
            int(len(df[df["y_true"] == clas]) * percent_missed),
        )

        df.loc[misclass, "y"] = rand.choice(
            [i for i in range(len(ys)) if i != int(clas)]
        )
        df.loc[misclass, "mislabeled"] = True
    return df


def mislabel_pairs(df, pairs, percent_missed):
    _, class_dict = get_classes(df)
    df["y_true"] = df["y"]
    df["mislabeled"] = False

    for source in pairs:
        src_amt = int(
            len(df[df["Species_Name"] == source]) / len(pairs[source]) * percent_missed
        )

        for target in pairs[source]:
            misclass_a = rand.sample(
                sorted(df.loc[df["Species_Name"] == target].index),
                int(len(df[df["Species_Name"] == target]) * percent_missed),
            )
            misclass_b = rand.sample(
                sorted(df.loc[df["Species_Name"] == source].index), src_amt
            )
            df.loc[misclass_a, "y"] = class_dict[source]
            df.loc[misclass_a, "mislabeled"] = True
            df.loc[misclass_b, "y"] = class_dict[target]
            df.loc[misclass_b, "mislabeled"] = True

    return df


def test_model(settings_json):
    class_dict, _ = get_classes(get_data_csv(settings_json["data_csv_path"]))
    settings_json["num_classes"] = len(class_dict)
    settings_json["model_kwargs"]["num_classes"] = len(class_dict)
    print(settings_json)
    config = ExperimentationConfig.parse_obj(settings_json)
    start = time.time()
    orsac_results = OrsacResults(config, threshold=1.0, test_mode=True)
    orsac_results.save_figures()
    orsac_results.save_results_df()
#Skip top_x for now, consider removing. 
    # orsac_results.save_top_x_figs()

    end = time.time()
    print(f"Time elapsed:{start-end}")


def orsac_test(experiment_dir: str):
    with open(experiment_dir + "/config.json", "r") as json_file:
        settings_json = json.load(json_file)
        settings_json["mode"] = "test"  # Change the mode to "test"
        print(settings_json["mode"])
        test_model(settings_json)
