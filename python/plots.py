from functools import reduce
from pprint import pprint
import seaborn.timeseries
import re
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob

impls = [
    # "cudnn",
    "libtorch",
    "pytorch",
]
datasets = [
    "mnist",
    "cifar10",
    # "stl10",
    "pascal",
]


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


# def _plot_range_band(*args, central_data=None, ci=None, data=None, **kwargs):
#     upper = data.max(axis=0)
#     lower = data.min(axis=0)
#     #import pdb; pdb.set_trace()
#     ci = np.asarray((lower, upper))
#     kwargs.update({"central_data": central_data, "ci": ci, "data": data})
#     seaborn.timeseries._plot_ci_band(*args, **kwargs)
#
# seaborn.timeseries._plot_range_band = _plot_range_band


def plot(
    min,
    mean,
    max,
    fig_ax=None,
    label="cudnn",
    title="Average train loss per epoch",
    ylabel="loss",
    shift=np.abs(np.random.normal(0, 1e-7, 100)),
):
    fig, ax = fig_ax
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    x = range(len(mean))
    ax.plot(x, mean + shift, "-", label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    ax.fill_between(x, min + shift, max + shift, alpha=0.2)

    return fig, ax


def get_min_mean_max(dfs, impl, dataset, train=True):
    grouped = dfs[impl][dataset]["train" if train else "eval"].groupby(level="epoch")
    return (grouped.min(), grouped.mean(), grouped.max())


if __name__ == "__main__":
    # cleanup_profiles()
    dfs = make_dfs()
    key = "avg loss"
    fig = ax = None
    train = True
    for dataset in datasets:
        fig = ax = None
        for impl in impls:
            min, mean, max = get_min_mean_max(dfs, impl, dataset, train=train)
            fig, ax = plot(
                min[key].values,
                mean[key].values,
                max[key].values,
                (fig, ax),
                label=impl,
                title=f"{'train' if train else 'eval'} {key} per epoch {dataset}",
            )
        plt.show()
    # print(min)
    # y = dfs["cudnn"]["cifar10"]["train"].loc[0]["avg loss"]
    # plot()
