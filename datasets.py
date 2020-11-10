from torchvision.datasets import FashionMNIST, STL10, VOCDetection, MNIST

mnist_train = MNIST("data", train=True, download=True)
mnist_test = MNIST("data", train=False, download=True)

fashion_mnist_train = FashionMNIST("data", train=True, download=True)
fashion_mnist_test = FashionMNIST("data", train=False, download=True)

stl_10_train = STL10("data", split="train", download=True)
stl_10_test = STL10("data", split="test", download=True)

pascal_train = VOCDetection("data", year="2012", image_set="train", download=True)
pascal_test = VOCDetection("data", year="2012", image_set="val", download=True)
