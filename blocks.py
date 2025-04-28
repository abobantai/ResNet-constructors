import torch.nn as nn
#default
class ResBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, koef=None, act1 = None, act2 = None):
        
        super(ResBlockV1, self).__init__()
        if act1 is None:
            act1 = nn.ReLU(inplace=True)
        if act2 is None:
            act2 = act1
        if stride is None:
            stride = 1
        if koef is None:
            koef = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act1
        self.act2 = act2
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
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * self.koef
        out += self.shortcut(x)
        out = self.act2(out)
        return out

#without bn2
class ResBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, koef=None, act1 = None, act2 = None):
        super(ResBlockV2, self).__init__()
        if act1 is None:
            act1 = nn.ReLU(inplace=True)
        if act2 is None:
            act2 = act1
        if stride is None:
            stride = 1
        if koef is None:
            koef = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act1
        self.act2 = act2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
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
        out = self.act1(out)
        out = self.conv2(out)
        out = out * self.koef
        out += self.shortcut(x)
        out = self.act2(out)
        return out

#act before shortcut
class ResBlockV3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, koef=None, act1 = None, act2 = None):
        super(ResBlockV3, self).__init__()
        if act1 is None:
            act1 = nn.ReLU(inplace=True)
        if act2 is None:
            act2 = act1
        if stride is None:
            stride = 1
        if koef is None:
            koef = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act1
        self.act2 = act2
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
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * self.koef
        out = self.act2(out)
        out += self.shortcut(x)
        return out

#without bn2 and act before shortcut
class ResBlockV4(nn.Module):

    def __init__(self, in_channels, out_channels, stride=None, koef=None, act1 = None, act2 = None):
        super(ResBlockV4, self).__init__()
        if act1 is None:
            act1 = nn.ReLU(inplace=True)
        if act2 is None:
            act2 = act1
        if stride is None:
            stride = 1
        if koef is None:
            koef = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act1
        self.act2 = act2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
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
        out = self.act1(out)
        out = self.conv2(out)
        out = out * self.koef
        out = self.act2(out)
        out += self.shortcut(x)
        return out

#without act
class ResBlockV5(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, koef=None, act1 = None, act2 = None):
        super(ResBlockV5, self).__init__()
        if act1 is None:
            act1 = nn.ReLU(inplace=True)
        if act2 is None:
            act2 = act1
        if stride is None:
            stride = 1
        if koef is None:
            koef = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act1
        self.act2 = act2
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
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * self.koef
        out += self.shortcut(x)
        return out

#without bn2 and without act 
class ResBlockV6(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, koef=None, act1 = None, act2 = None):
        super(ResBlockV6, self).__init__()
        if act1 is None:
            act1 = nn.ReLU(inplace=True)
        if act2 is None:
            act2 = act1
        if stride is None:
            stride = 1
        if koef is None:
            koef = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act1
        self.act2 = act2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
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
        out = self.act1(out)
        out = self.conv2(out)
        out = out * self.koef
        out += self.shortcut(x)
        return out
