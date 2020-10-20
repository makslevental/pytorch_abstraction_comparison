import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50')
model.eval()
print(model)
