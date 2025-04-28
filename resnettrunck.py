import torch.nn as nn
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, koef=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.koef = koef
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * self.koef

        out += self.shortcut(x)
        out = self.relu(out)

        return out
class ResTruck(nn.Module):
    def __init__(self, block=None, strides=None, num_blocks=None, channels=None, koef=None):
        super(ResTruck, self).__init__()
        self.stages = self._make_layers(
            block=block,
            strides=strides,
            num_blocks=num_blocks,
            channels=channels,
            koef=koef
        )


        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))





    def _make_layers(self, num_blocks = None, block = None, strides = None, channels = None, koef = None):
#        """
#        set default values for strides, channels, koef
#        """
        

        if strides is None:
            strides = [1] * len(num_blocks)
        if channels is None:
            channels = []
            for i in range(len(num_blocks)):
                channels.append(16 * (2 ** i))
        if koef is None:
            koef = [1] * len(num_blocks)
        if block is None:
            block = ResNetBlock




#       handling exceptional situations
        if num_blocks is None:
            raise ValueError("num_blocks must be provided")
        if len(num_blocks) != len(channels):
            raise ValueError("num_blocks and channels must have the same length")
        if len(koef) != len(num_blocks):
            raise ValueError("num_blocks and koef must have the same length")
        if len(strides) != len(num_blocks):
            raise ValueError("num_blocks and strides must have the same length")
        
        

#       create the layers
        stages = []
        for i in range(len(num_blocks)):
            blocks = []
            for j in range(num_blocks[i]):
                if j != num_blocks[i] - 1:
                    if j == 0:
                        blocks.append(block(channels[i], channels[i], stride=strides[i], koef = koef[i]))
                    else:
                        blocks.append(block(channels[i], channels[i], stride=1, koef=koef[i]))
                else:
                    if i != len(num_blocks)-1:
                        blocks.append(block(channels[i], channels[i+1], stride=1, koef=koef[i]))
                    else:
                        blocks.append(block(channels[i], channels[i], stride=1, koef=koef[i]))
            stages.append(nn.Sequential(*blocks))
        return nn.Sequential(*stages)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.stages(out)
        out = self.avgpool(out)
        return out
class Resnet_construct(nn.Module):
    def __init__(self, block, strides, num_blocks, channels, koef=None, num_classes=100):
        super(Resnet_construct, self).__init__()
        self.model = ResTruck(block=block, strides=strides, num_blocks=num_blocks, channels=channels, koef=koef)
        self.flatten = nn.Flatten()
        if channels is None:
            channels = [16 * (2 ** i) for i in range(len(num_blocks))]
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        out = self.model(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    

def create(block=ResNetBlock, strides=None, num_blocks=None, channels=None, koef=None, num_classes=100):
    return Resnet_construct(block=block, strides=strides, num_blocks=num_blocks, channels=channels, koef=koef, num_classes=num_classes)

print("\033[31mResNetTrunk loaded\033[34m")
print("func create have:")
print("\tblock: name of blocks default(ResNetBlock)")
print("\tstrides: list of integers, default([1]*len(num_blocks))")
print("\tnum_blocks: list of integers")
print("\tchannels: list of integers, default([16*(2**i) for i in range(len(num_blocks))])")
print("\tkoef: list of integers, default([1]*len(num_blocks))")
print("\tnum_classes: integer, default(100)\033[31m")
print("ResNetTrunk loaded\033[0m")


