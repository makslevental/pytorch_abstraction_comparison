import os
import sys
import xml.etree.ElementTree as ET
import py3nvml.py3nvml as nvml
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.cuda.nvtx import range_push, range_pop

from cuda_profiling import (
    GPUTimer,
    get_used_cuda_mem,
    get_gpu_utilization,
    cuda_profiler_start,
)
from pascal import VOCDetection
from resnet import ResNet50

DEVICE = int(os.environ["DEVICE"])
print(f"device {DEVICE}")
gpu_timer = GPUTimer(DEVICE)

nvml.nvmlInit()
nvml_handle = nvml.nvmlDeviceGetHandleByIndex(DEVICE)


def train(
    model,
    trainloader,
    testloader,
    optimizer,
    criterion,
    epochs,
    batch_size,
    monitoring_step,
    output_file=None,
):
    cuda_profiler_start()
    for epoch in range(epochs):
        model.train()

        running_elapsed_time = loss_val = running_loss = 0
        elapsed_time = (
            running_sample_count
        ) = tp_count = running_tp_count = sample_count = 0
        running_used_mem = used_mem = 0

        for batch_n, (inputs, labels) in enumerate(trainloader, 0):
            range_push(f"train epoch {epoch} batch {batch_n}")
            range_push(f"batch load")
            gpu_timer.start()

            optimizer.zero_grad()

            inputs = inputs.to(f"cuda:{DEVICE}")
            labels = labels.to(f"cuda:{DEVICE}")
            range_pop()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            range_pop()

            gpu_timer.stop()

            loss_val += loss.item()
            predicted = outputs.argmax(axis=1)
            tp_count += (predicted == labels).sum().item()
            sample_count += batch_size
            elapsed_time += gpu_timer.elapsed_time()
            used_mem += get_used_cuda_mem(DEVICE)

            if (batch_n + 1) % monitoring_step == 0:
                print(f"batch: {batch_n}")
                print(
                    f"[TRAIN] epoch: {epoch}, "
                    f"avg loss: {loss_val / float(sample_count):.10f}, "
                    f"accuracy: {100.0 * float(tp_count) / sample_count:.6f}%, "
                    f"avg sample time: {elapsed_time / sample_count:.6f}ms, "
                    f"avg used mem: {used_mem / monitoring_step:.6f}mb, "
                    f"avg util rate: {get_gpu_utilization(nvml_handle)}%",
                    file=output_file,
                    flush=True,
                )
                running_elapsed_time += elapsed_time
                running_loss += loss_val
                running_tp_count += tp_count
                running_sample_count += sample_count
                running_used_mem += used_mem
                used_mem = elapsed_time = tp_count = sample_count = loss_val = 0

        print(
            f"[TRAIN SUMMARY] "
            f"avg loss: {running_loss / float(running_sample_count):.10f}, "
            f"accuracy: {100.0 * float(running_tp_count) / running_sample_count:.6f}%, "
            f"avg sample time: {running_elapsed_time / running_sample_count:.6f}ms, "
            f"avg used mem: {running_used_mem / (running_sample_count / batch_size)}mb, "
            f"avg gpu util: {get_gpu_utilization(nvml_handle)}%",
            file=output_file,
            flush=True,
        )

        model.eval()
        sample_count = tp_count = loss_val = 0
        used_mem = 0

        with torch.no_grad():
            for batch_n, (inputs, labels) in enumerate(testloader):
                range_push(f"eval epoch {epoch} batch {batch_n}")
                range_push(f"batch load")
                gpu_timer.start()

                inputs = inputs.to(f"cuda:{DEVICE}")
                labels = labels.to(f"cuda:{DEVICE}")

                range_pop()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                range_pop()

                gpu_timer.stop()

                loss_val += loss.item()
                tp_count += (predicted == labels).sum().item()
                sample_count += batch_size
                elapsed_time += gpu_timer.elapsed_time()
                used_mem += get_used_cuda_mem(DEVICE)

        print(
            f"[EVAL] "
            f"avg loss: {loss_val / float(sample_count):.10f}, "
            f"accuracy: {100.0 * float(tp_count) / sample_count:.6f}%, "
            f"avg sample time: {elapsed_time / sample_count:.6f}ms, "
            f"avg used mem: {used_mem / (sample_count / batch_size)}mb, "
            f"avg gpu util: {get_gpu_utilization(nvml_handle)}%",
            file=output_file,
            flush=True,
        )


object_categories = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

object_categories_idx = dict(
    list(zip(object_categories, range(len(object_categories))))
)


def transform_pascal(x):
    name = x["annotation"]["object"][0]["name"]
    idx = object_categories_idx[name]
    return idx
    # k = torch.zeros(len(object_categories), dtype=torch.long)
    # k[idx] = 1
    # return k


def main():
    dataset_name, run_n, epochs, resolution = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
    )
    print(f"running {dataset_name} {run_n}")
    print(epochs, resolution)
    # lost .5 ms with smaller batch size for mnist
    batch_size = 128
    monitoring_step = 20

    transform = [
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="../data",
            train=True,
            download=True,
            transform=transforms.Compose(transform),
        )
        testset = torchvision.datasets.CIFAR10(
            root="../data",
            train=False,
            download=True,
            transform=transforms.Compose(transform),
        )
        in_channels = 3
        model = ResNet50(in_channels=in_channels, num_classes=10)
    elif dataset_name == "stl10":
        trainset = torchvision.datasets.STL10(
            root="../data",
            split="train",
            download=True,
            transform=transforms.Compose(transform),
        )
        testset = torchvision.datasets.STL10(
            root="../data",
            split="test",
            download=True,
            transform=transforms.Compose(transform),
        )
        in_channels = 3
        model = ResNet50(in_channels=in_channels, num_classes=10)
    elif dataset_name == "mnist":
        trainset = torchvision.datasets.MNIST(
            root="../data",
            train=True,
            download=True,
            transform=transforms.Compose(transform),
        )
        testset = torchvision.datasets.MNIST(
            root="../data",
            train=False,
            download=True,
            transform=transforms.Compose(transform),
        )
        in_channels = 1
        model = ResNet50(in_channels=in_channels, num_classes=10)
    elif dataset_name == "pascal":
        batch_size = 32
        transform.insert(0, transforms.Resize((resolution, resolution)))
        trainset = VOCDetection(
            root="../data/",
            year="2012",
            image_set="train",
            download=True,
            transform=transforms.Compose(transform),
            target_transform=transform_pascal,
        )
        print("len trainset", len(trainset))
        testset = VOCDetection(
            root="../data/",
            year="2012",
            image_set="val",
            download=True,
            transform=transforms.Compose(transform),
            target_transform=transform_pascal,
        )
        print("len testset", len(testset))
        in_channels = 3
        model = ResNet50(in_channels=in_channels, num_classes=20)
    else:
        raise Exception("unsupported dataset")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    model = model.to(f"cuda:{DEVICE}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(
        model,
        trainloader,
        testloader,
        optimizer,
        criterion,
        epochs,
        batch_size,
        monitoring_step,
        open(
            f"profiles/resolution/run_pytorch_{dataset_name}_{run_n}_{resolution}.csv",
            "w",
        ),
    )


if __name__ == "__main__":
    main()
