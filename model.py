import resnettrunck as rt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

strides = [1, 2, 2]
num_blocks = [2, 2, 2]
channels = [16, 32, 64]
koef = [1, 1, 1]
num_classes = 10

model = rt.create(
    block = rt.ResNetBlock, 
    strides = strides, 
    num_blocks = num_blocks, 
    channels = channels, 
    koef=koef, 
    num_classes=100
).to(device)

print(model)
