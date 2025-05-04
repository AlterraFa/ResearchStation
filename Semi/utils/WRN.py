import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float = 0.0):
        super(ResnetBlock, self).__init__()
        
        self.dropout = dropout
        
        self.bn1 = nn.BatchNorm2d(in_channels, momentum = 0.01)
        self.LeakyReLU1 = nn.LeakyReLU(inplace = True, negative_slope = 0.01)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum = 0.01)
        self.LeakyReLU2 = nn.LeakyReLU(inplace = True, negative_slope = 0.01)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.InIsOut = in_channels == out_channels
        self.shortcutCompat = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0, bias = False) if not self.InIsOut else nn.Identity() 
        
    def forward(self, x):
        if not self.InIsOut:
            x = self.LeakyReLU1(self.bn1(x))
        else:
            out = self.LeakyReLU1(self.bn1(x))

        
        out = self.LeakyReLU2(self.bn2(self.conv1(out if self.InIsOut else x)))
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, training = self.training)
            
        out = self.conv2(out)
        
        return self.shortcutCompat(x) + out

class BlockStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float, numBlock: int):
        super(BlockStack, self).__init__() 
        self.group = self.make_group(in_channels, out_channels, stride, dropout, numBlock)
        
    def make_group(self, in_channels: int, out_channels: int, stride: int, dropout: float, numBlock: int):
        layers = []
        for idx in range(numBlock):
            if idx == 0:
                layers.append(ResnetBlock(in_channels, out_channels, stride, dropout))
            else:
                layers.append(ResnetBlock(out_channels, out_channels, 1, dropout))
        
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.group(x)

        

class WRN(nn.Module):
    def __init__(self, depth: int, widenFact: int, numClasses: int, dropout: float = 0.0):
        super(WRN, self).__init__()
        assert (depth - 4) % 6 == 0
        numBlock = (depth - 4) // 6

        channelDepth = [16, 16 * widenFact, 32 * widenFact, 64 * widenFact]
        strides = [1, 2, 2]

        self.stem = nn.Conv2d(3, channelDepth[0], kernel_size = 3, stride = 1)
        
        self.largeGroup = nn.ModuleList(
            [BlockStack(channelDepth[i], channelDepth[i + 1], strides[i], dropout, numBlock) for i in range(3)]
        )

        self.bn = nn.BatchNorm2d(channelDepth[-1], momentum = 0.01)
        self.LeakyReLU = nn.LeakyReLU(inplace = True, negative_slope = 0.01)
        self.fc = nn.LazyLinear(numClasses)

        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        
    def forward(self, x):
        x = self.stem(x)
        for group in self.largeGroup:
            x = group(x)
        x = self.LeakyReLU(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def summary(self):
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_MB = total_params * 4 / (1024 ** 2)  # Assuming 32-bit float = 4 bytes
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"Approximate Model Size: {total_MB:.2f} MB")

