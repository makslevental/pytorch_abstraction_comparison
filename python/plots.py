import csv
import glob
import os
import re
from collections import defaultdict
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401 unused import

plt.style.use("ggplot")

impls = ["cudnn", "libtorch", "pytorch"]
datasets = ["mnist", "cifar10", "stl10", "pascal"]


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def cleanup_profiles_cudnn():
    for impl in ["cudnn"]:
        for dataset in datasets:
            train_summaries = list(
                reduce(
                    lambda accum, fp: accum
                    + [
                        f"{fp[0]}," + f"{epoch}," + re.sub(r"[a-zA-Z%\[\]:\s]", "", l)
                        for (epoch, l) in enumerate(open(fp[1]).readlines())
                        if "SUMMARY" in l
                    ],
                    enumerate(
                        sorted(glob.glob(f"profiles/cudnn2/run*{impl}_{dataset}_*.csv"))
                    ),
                    [],
                )
            )
            eval_summaries = list(
                reduce(
                    lambda accum, fp: accum
                    + [
                        f"{fp[0]}," + f"{epoch}," + re.sub(r"[a-zA-Z%\[\]:\s]", "", l)
                        for (epoch, l) in enumerate(open(fp[1]).readlines())
                        if "EVAL" in l
                    ],
                    enumerate(
                        sorted(glob.glob(f"profiles/cudnn2/run*{impl}_{dataset}_*.csv"))
                    ),
                    [],
                )
            )
            with open(f"profiles/summaries/train_{impl}_{dataset}.csv", "w") as csv:
                csv.write(
                    "run, epoch, avg loss, accuracy, avg sample time, avg used mem, avg gpu util\n"
                )
                csv.write("\n".join(train_summaries))
            with open(f"profiles/summaries/eval_{impl}_{dataset}.csv", "w") as csv:
                csv.write(
                    "run, epoch, avg loss, accuracy, avg sample time, avg used mem, avg gpu util\n"
                )
                csv.write("\n".join(eval_summaries))


def cleanup_profiles():
    for impl in impls:
        for dataset in datasets:
            train_summaries = list(
                reduce(
                    lambda accum, fp: accum
                    + [
                        f"{fp[0]}," + f"{epoch}," + re.sub(r"[a-zA-Z%\[\]:\s]", "", l)
                        for (epoch, l) in enumerate(open(fp[1]).readlines())
                        if "SUMMARY" in l
                    ],
                    enumerate(
                        sorted(glob.glob(f"profiles/run*{impl}_{dataset}_*.csv"))
                    ),
                    [],
                )
            )
            eval_summaries = list(
                reduce(
                    lambda accum, fp: accum
                    + [
                        f"{fp[0]}," + f"{epoch}," + re.sub(r"[a-zA-Z%\[\]:\s]", "", l)
                        for (epoch, l) in enumerate(open(fp[1]).readlines())
                        if "EVAL" in l
                    ],
                    enumerate(
                        sorted(glob.glob(f"profiles/run*{impl}_{dataset}_*.csv"))
                    ),
                    [],
                )
            )
            with open(f"profiles/summaries/train_{impl}_{dataset}.csv", "w") as csv:
                csv.write(
                    "run, epoch, avg loss, accuracy, avg sample time, avg used mem, avg gpu util\n"
                )
                csv.write("\n".join(train_summaries))
            with open(f"profiles/summaries/eval_{impl}_{dataset}.csv", "w") as csv:
                csv.write(
                    "run, epoch, avg loss, accuracy, avg sample time, avg used mem, avg gpu util\n"
                )
                csv.write("\n".join(eval_summaries))


def cleanup_resolutions():
    for impl in ["cudnn", "libtorch", "pytorch", "torchscript"]:
        for dataset in ["pascal"]:
            for batch_size in range(3, 9 + 1):
                for resolution in range(3, 12 + 1):
                    fp = f"profiles/resolution/run_{impl}_{dataset}_{2 ** batch_size}_{2 ** resolution}.csv"
                    fp_clean_train = f"profiles/resolution/clean_train_run_{impl}_{dataset}_{2 ** batch_size}_{2 ** resolution}.csv"
                    fp_clean_eval = f"profiles/resolution/clean_eval_run_{impl}_{dataset}_{2 ** batch_size}_{2 ** resolution}.csv"
                    if os.path.exists(fp):
                        with open(fp, "r") as src:
                            lines = src.readlines()
                            if len(lines) > 1:
                                with open(fp_clean_train, "w") as dst1, open(
                                    fp_clean_eval, "w"
                                ) as dst2:
                                    dst1.write(
                                        "avg loss, accuracy, avg sample time, avg used mem, avg gpu util\n"
                                    )
                                    dst2.write(
                                        "avg loss, accuracy, avg sample time, avg used mem, avg gpu util\n"
                                    )
                                    for l in lines:
                                        ll = re.sub(r"[a-zA-Z%\[\]:\s]", "", l)
                                        if "[TRAIN SUMMARY]" in l:
                                            dst1.write(f"{ll}\n")
                                        elif "[EVAL]" in l:
                                            dst2.write(f"{ll}\n")


def make_resolution_dfs():
    runs = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict)))
    )

    for fp in sorted(glob.glob(f"profiles/resolution/*.csv")):
        if len(open(fp).readlines()) > 1:
            df = pd.read_csv(fp, " *, *", engine="python")
            match = re.match(
                r".*/clean_(?P<train>\w+)_run_(?P<impl>\w+)_pascal_(?P<batch_size>\w+)_(?P<resolution>\w+).csv",
                fp,
            )
            if match:
                m = match.groupdict()
                runs[m["impl"]][m["train"]][int(m["batch_size"])][
                    int(m["resolution"])
                ] = df[["avg sample time", "avg used mem", "avg gpu util"]]

    return runs


def make_dfs():
    runs = defaultdict(lambda: defaultdict(dict))

    for impl in impls:
        for dataset in datasets:
            runs[impl][dataset]["train"] = pd.read_csv(
                f"profiles/summaries/train_{impl}_{dataset}.csv",
                " *, *",
                engine="python",
            ).set_index(["run", "epoch"])
            runs[impl][dataset]["eval"] = pd.read_csv(
                f"profiles/summaries/eval_{impl}_{dataset}.csv",
                " *, *",
                engine="python",
            ).set_index(["run", "epoch"])

    return runs


def plot(
    min,
    mean,
    max,
    times,
    fig_ax=None,
    label="cudnn",
    title="Average train loss per epoch",
    ylabel="loss",
    shift=np.abs(np.random.normal(0, 1e-70, 100)),
):
    fig, ax = fig_ax
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.plot(times, mean + shift, "-", label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch/time")
    ax.set_yscale("log")
    ax.fill_between(times, min + shift, max + shift, alpha=0.2)

    return fig, ax


def scatter(
    min,
    mean,
    max,
    times,
    fig_ax=None,
    label="cudnn",
    title="Average train loss per epoch",
    ylabel="loss",
    xlabel="",
    shift=np.abs(np.random.normal(0, 1e-70, 100)),
):
    times = np.log2(times)
    fig, ax = fig_ax
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.plot(times, mean + shift, "-", label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yscale("log")
    ax.fill_between(times, min + shift, max + shift, alpha=0.2)

    return fig, ax


def get_min_mean_max(dfs, impl, dataset, train=True):
    grouped = dfs[impl][dataset]["train" if train else "eval"].groupby(level="epoch")
    return (grouped.min(), grouped.mean(), grouped.max())


def plot_all(profile_dfs, resolution_dfs: dict):
    impls = ["cudnn", "libtorch", "pytorch", "torchscript"]
    input_list = []
    for key in ["accuracy", "avg loss"]:
        for train in [True, False]:
            for dataset in datasets:
                fig = ax = None
                for impl in impls:
                    if ("train" if train else "eval") not in profile_dfs[impl][dataset]:
                        continue
                    min, mean, max = get_min_mean_max(
                        profile_dfs, impl, dataset, train=train
                    )
                    fig, ax = plot(
                        min[key].values,
                        mean[key].values,
                        max[key].values,
                        mean["avg sample time"].values.cumsum() if train else list(range(100)),
                        (fig, ax),
                        label=impl,
                        title=f"{'train' if train else 'eval'} {key} per epoch {dataset}",
                        ylabel=key,
                    )
                # plt.show()
                tikzplotlib.clean_figure()
                tfp = f"../tex/figures/plots/{'train' if train else 'eval'} {key} per epoch {dataset}.tex".replace(
                    " ", "_"
                )
                input_list.append(tfp)
                tikzplotlib.save(tfp, standalone=False)

    # units = {
    #     "avg sample time": "time (ms)",
    #     "avg used mem": "memory (mb)",
    #     "avg gpu util": "util (\%)",
    # }
    # batch_sizes = np.power(2, list(range(3, 9 + 1)))
    # resolutions = np.power(2, list(range(3, 12 + 1)))
    # for train in ["train", "eval"]:
    #     for key in ["avg sample time", "avg used mem", "avg gpu util"]:
    #         for impl in impls:
    #             if train not in resolution_dfs[impl]:
    #                 continue
    #
    #             points = []
    #             for i, batch_size in enumerate(batch_sizes):
    #                 for j, resolution in enumerate(resolutions):
    #                     if (
    #                         batch_size in resolution_dfs[impl][train]
    #                         and resolution in resolution_dfs[impl][train][batch_size]
    #                     ):
    #                         points.append(
    #                             (
    #                                 batch_size,
    #                                 resolution,
    #                                 resolution_dfs[impl][train][batch_size][resolution][
    #                                     key
    #                                 ].mean(),
    #                             )
    #                         )
    #             if points:
    #                 csv.writer(
    #                     open(
    #                         f"../tex/figures/plots/{impl}_{train}_{key}.csv".replace(
    #                             " ", "_"
    #                         ),
    #                         "w",
    #                     ),
    #                     delimiter=" ",
    #                 ).writerows(points)
    #         gfp = f"../tex/figures/plots/{train}_{key}.tex".replace(" ", "_")
    #         input_list.append(gfp)
    #         with open("../tex/figures/plots/scatter_tex") as f, open(gfp, "w") as g:
    #             text = replace_all(
    #                 f.read(),
    #                 {
    #                     "{z_label}": units[key],
    #                     "{title}": f"{key} on {train}",
    #                     "{cudnn_csv}": f"./cudnn_{train}_{key}.csv".replace(" ", "_"),
    #                     "{libtorch_csv}": f"./libtorch_{train}_{key}.csv".replace(
    #                         " ", "_"
    #                     ),
    #                     "{pytorch_csv}": f"./pytorch_{train}_{key}.csv".replace(
    #                         " ", "_"
    #                     ),
    #                     "{torchscript_csv}": f"./torchscript_{train}_{key}.csv".replace(
    #                         " ", "_"
    #                     ),
    #                 },
    #             )
    #             g.write(text)
    # units = {
    #     "avg sample time": "time (ms)",
    #     "avg used mem": "memory (mb)",
    #     "avg gpu util": "util (\%)",
    # }
    # batch_sizes = [32]
    # resolutions = np.power(2, list(range(3, 12 + 1)))
    # for train in ["train", "eval"]:
    #     for key in ["avg sample time", "avg used mem", "avg gpu util"]:
    #         fig = ax = None
    #         for impl in impls + ["torchscript"]:
    #             points = []
    #             for i, batch_size in enumerate(batch_sizes):
    #                 for j, resolution in enumerate(resolutions):
    #                     if (
    #                         batch_size in resolution_dfs[impl][train]
    #                         and resolution in resolution_dfs[impl][train][batch_size]
    #                     ):
    #                         points.append(
    #                             (
    #                                 batch_size,
    #                                 resolution,
    #                                 resolution_dfs[impl][train][batch_size][resolution][
    #                                     key
    #                                 ].values,
    #                             )
    #                         )
    #             if points:
    #                 fig, ax = scatter(
    #                     [x[2].min() for x in points],
    #                     [x[2].mean() for x in points],
    #                     [x[2].max() for x in points],
    #                     [x[1] for x in points],
    #                     (fig, ax),
    #                     label=impl,
    #                     title=f"batch size {batch_size} {key} for resolutions",
    #                     ylabel=units[key],
    #                     xlabel="resolution",
    #                     shift=np.abs(np.random.normal(0, 1e-70, len(points))),
    #                 )
    #
    #         # plt.show()
    #         tikzplotlib.clean_figure()
    #         tfp = f"../tex/figures/plots/batch size {batch_size} {key} for resolutions.tex".replace(
    #             " ", "_"
    #         )
    #         input_list.append(tfp)
    #         tikzplotlib.save(tfp, standalone=False)
    #
    # batch_sizes = np.power(2, list(range(3, 9 + 1)))
    # resolutions = [32]
    # for train in ["train", "eval"]:
    #     for key in ["avg sample time", "avg used mem", "avg gpu util"]:
    #         fig = ax = None
    #         for impl in impls + ["torchscript"]:
    #             points = []
    #             for i, batch_size in enumerate(batch_sizes):
    #                 for j, resolution in enumerate(resolutions):
    #                     if (
    #                         batch_size in resolution_dfs[impl][train]
    #                         and resolution in resolution_dfs[impl][train][batch_size]
    #                     ):
    #                         points.append(
    #                             (
    #                                 batch_size,
    #                                 resolution,
    #                                 resolution_dfs[impl][train][batch_size][resolution][
    #                                     key
    #                                 ].values,
    #                             )
    #                         )
    #             if points:
    #                 fig, ax = scatter(
    #                     [x[2].min() for x in points],
    #                     [x[2].mean() for x in points],
    #                     [x[2].max() for x in points],
    #                     [x[0] for x in points],
    #                     (fig, ax),
    #                     label=impl,
    #                     title=f"resolution {resolution} {key} for batch size",
    #                     ylabel=units[key],
    #                     xlabel="batch size",
    #                     shift=np.abs(np.random.normal(0, 1e-70, len(points))),
    #                 )
    #         # plt.show()
    #         tikzplotlib.clean_figure()
    #         tfp = f"../tex/figures/plots/resolution {resolution} {key} for batch size.tex".replace(
    #             " ", "_"
    #         )
    #         input_list.append(tfp)
    #         tikzplotlib.save(tfp, standalone=False)
    #
    #
    # u = """
    # \\begin{{figure}}
    #     {tikz}
    # \end{{figure}}
    # """
    # with open("../tex/figures/plots/all_plots_tex") as tf, open(
    #     "../tex/figures/plots/all_plots.tex", "w"
    # ) as uf:
    #     text = tf.read()
    #     uf.write(
    #         text.replace(
    #             "{input_list}",
    #             "\n".join(
    #                 [
    #                     u.format(tikz=f"\input{{{i}}}".replace("../tex/", ""))
    #                     for i in input_list
    #                 ]
    #             ),
    #         )
    #     )


if __name__ == "__main__":
    # cleanup_profiles()
    cleanup_profiles_cudnn()
    # cleanup_resolutions()
    profile_dfs = make_dfs()
    resolution_dfs = make_resolution_dfs()
    plot_all(profile_dfs, resolution_dfs)
