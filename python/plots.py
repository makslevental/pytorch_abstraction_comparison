import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cudnn = pd.read_csv("profiles/run_cudnn_cifar10_0.csv")
libtorch = pd.read_csv("profiles/run_libtorch_cifar10_0.csv")
pytorch = pd.read_csv("profiles/run_pytorch_cifar10_0.csv")


def get_loss(df, word="TRAIN"):
    rows = df.iloc[:, 0].str.contains(f"\[{word}\] avg loss")
    avg_loss = (
        df[rows]
            .iloc[:, 0]
            .apply(lambda x: float(x.replace(f"[{word}] avg loss: ", "")))
    )
    return avg_loss


def get_accuracy(df, word="TRAIN"):
    rows = df.iloc[:, 0].str.contains(f"\[{word}\] avg loss")
    accuracy = (
        df[rows]
            .iloc[:, 1]
            .apply(lambda x: float(x.replace(f"accuracy: ", "").replace("%", "")))
    )
    return accuracy


def get_sample_time(df, word="TRAIN"):
    rows = df.iloc[:, 0].str.contains(f"\[{word}\] avg loss")
    sample_time = (
        df[rows]
            .iloc[:, 2]
            .apply(lambda x: float(x.replace(f"avg sample time: ", "").replace("ms", "")))
    )
    return sample_time


cudnn_train_loss = get_loss(cudnn, "TRAIN")
cudnn_eval_loss = get_loss(cudnn, "EVAL")
cudnn_train_accuracy = get_accuracy(cudnn, "TRAIN")
cudnn_eval_accuracy = get_accuracy(cudnn, "EVAL")
cudnn_train_sample_time = get_sample_time(cudnn, "TRAIN")
cudnn_eval_sample_time = get_sample_time(cudnn, "EVAL")

libtorch_train_loss = get_loss(libtorch, "TRAIN")
libtorch_eval_loss = get_loss(libtorch, "EVAL")
libtorch_train_accuracy = get_accuracy(libtorch, "TRAIN")
libtorch_eval_accuracy = get_accuracy(libtorch, "EVAL")
libtorch_train_sample_time = get_sample_time(libtorch, "TRAIN")
libtorch_eval_sample_time = get_sample_time(libtorch, "EVAL")

pytorch_train_loss = get_loss(pytorch, "TRAIN")
pytorch_eval_loss = get_loss(pytorch, "EVAL")
pytorch_train_accuracy = get_accuracy(pytorch, "TRAIN")
pytorch_eval_accuracy = get_accuracy(pytorch, "EVAL")
pytorch_train_sample_time = get_sample_time(pytorch, "TRAIN")
pytorch_eval_sample_time = get_sample_time(pytorch, "EVAL")

splot = sns.lineplot(x=range(len(cudnn_train_loss)), y=cudnn_train_loss, label="cudnn", legend=True)
splot = sns.lineplot(x=range(len(libtorch_train_loss)), y=libtorch_train_loss, label="libtorch", legend=True)
splot = sns.lineplot(x=range(len(pytorch_train_loss)), y=pytorch_train_loss, label="pytorch", legend=True)
splot.set(title="Average train loss per epoch")
splot.set(xlabel="epoch")
splot.set(ylabel="loss")
splot.set(yscale="log")
plt.show()

splot = sns.lineplot(x=range(len(cudnn_eval_loss)), y=cudnn_eval_loss, label="cudnn", legend=True)
splot = sns.lineplot(x=range(len(libtorch_eval_loss)), y=libtorch_eval_loss, label="libtorch", legend=True)
splot = sns.lineplot(x=range(len(pytorch_eval_loss)), y=pytorch_eval_loss, label="pytorch", legend=True)
splot.set(title="Average eval loss per epoch")
splot.set(xlabel="epoch")
splot.set(ylabel="loss")
splot.set(yscale="log")
plt.show()







splot = sns.lineplot(x=range(len(cudnn_train_accuracy)), y=cudnn_train_accuracy, label="cudnn", legend=True)
splot = sns.lineplot(x=range(len(libtorch_train_accuracy)), y=libtorch_train_accuracy, label="libtorch", legend=True)
splot = sns.lineplot(x=range(len(pytorch_train_accuracy)), y=pytorch_train_accuracy, label="pytorch", legend=True)
splot.set(title="Average train accuracy per epoch")
splot.set(xlabel="epoch")
splot.set(ylabel="accuracy")
splot.set(yscale="log")
plt.show()

splot = sns.lineplot(x=range(len(cudnn_eval_accuracy)), y=cudnn_eval_accuracy, label="cudnn", legend=True)
splot = sns.lineplot(x=range(len(libtorch_eval_accuracy)), y=libtorch_eval_accuracy, label="libtorch", legend=True)
splot = sns.lineplot(x=range(len(pytorch_eval_accuracy)), y=pytorch_eval_accuracy, label="pytorch", legend=True)
splot.set(title="Average eval accuracy per epoch")
splot.set(xlabel="epoch")
splot.set(ylabel="accuracy")
splot.set(yscale="log")
plt.show()

splot = sns.lineplot(x=range(len(cudnn_train_sample_time)), y=cudnn_train_sample_time, label="cudnn", legend=True)
splot = sns.lineplot(x=range(len(libtorch_train_sample_time)), y=libtorch_train_sample_time, label="libtorch", legend=True)
splot = sns.lineplot(x=range(len(pytorch_train_sample_time)), y=pytorch_train_sample_time, label="pytorch", legend=True)
splot.set(title="Average train sample_time per epoch")
splot.set(xlabel="epoch")
splot.set(ylabel="sample_time")
splot.set(yscale="log")
plt.show()

splot = sns.lineplot(x=range(len(cudnn_eval_sample_time)), y=cudnn_eval_sample_time, label="cudnn", legend=True)
splot = sns.lineplot(x=range(len(libtorch_eval_sample_time)), y=libtorch_eval_sample_time, label="libtorch", legend=True)
splot = sns.lineplot(x=range(len(pytorch_eval_sample_time)), y=pytorch_eval_sample_time, label="pytorch", legend=True)
splot.set(title="Average eval sample_time per epoch")
splot.set(xlabel="epoch")
splot.set(ylabel="sample_time")
splot.set(yscale="log")
plt.show()
