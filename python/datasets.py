import os
import numpy as np

from torchvision.datasets import FashionMNIST, STL10, VOCDetection, MNIST

mnist_train = MNIST("../data", train=True, download=True)
mnist_test = MNIST("../data", train=False, download=True)

fashion_mnist_train = FashionMNIST("../data", train=True, download=True)
fashion_mnist_test = FashionMNIST("../data", train=False, download=True)

stl_10_train = STL10("../data", split="train", download=True)
if not os.path.exists("../data/stl_10_train_data.npy"):
    np.save("../data/stl_10_train_data.npy", stl_10_train.data)
if not os.path.exists("../label/stl_10_train_labels.npy"):
    np.save("../data/stl_10_train_labels.npy", stl_10_train.labels)
stl_10_test = STL10("../data", split="test", download=True)
if not os.path.exists("../data/stl_10_test_data.npy"):
    np.save("../data/stl_10_test_data.npy", stl_10_test.data)
if not os.path.exists("../label/stl_10_test_labels.npy"):
    np.save("../data/stl_10_test_labels.npy", stl_10_test.labels)

pascal_train = VOCDetection("../data", year="2012", image_set="train", download=True)
pascal_test = VOCDetection("../data", year="2012", image_set="val", download=True)
