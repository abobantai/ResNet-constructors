import resnettrunck as rt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

strides = [1, 2, 2]
num_blocks = [2, 2, 2]
channels = [16, 32, 64]
koef = [1, 1, 1]
num_classes = 10

model = rt.create(
    block = rt.ResBlockV1, 
    strides = strides, 
    num_blocks = num_blocks, 
    channels = channels, 
    koef=koef, 
    num_classes=100,
    act1=torch.nn.ReLU(inplace=True),
    act2=torch.nn.LeakyReLU(inplace=True, negative_slope=0.01),
).to(device)

print(model)
