import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.layers.drop import DropPath

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float = 0.0): 
        # Adding group conv won't work well since there's no mixing mechanism in this
        # I need seperate Resnext block with a 1x1->3x3->1x1 not a 3x3->3x3
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
    

class ConvneXt(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float = 0.0, scaleInit: float = 1e-6):
        super(ConvneXt, self).__init__()
        
        if in_channels != out_channels or stride != 1:
            self.reduction = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = stride, bias = False),
                Rearrange("b c h w -> b h w c"),
                nn.LayerNorm(out_channels),
                Rearrange("b h w c -> b c h w"),
            )
        else:
            self.reduction = nn.Identity()

        self.compute    = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 7, stride = 1, padding = 3, groups = out_channels, bias = True),
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels),
            Rearrange("b h w c -> b c h w"),
        )
        self.layerScale = nn.Parameter(scaleInit * torch.ones(out_channels), 
                                        requires_grad=True)
        self.dropPath   = DropPath(dropout)
            
    def forward(self, x: torch.Tensor):
        x = self.reduction(x)
        
        out = self.compute(x)
        out = self.layerScale.view(1, -1, 1, 1) * out
        
        out = self.dropPath(out)
        
        return x + out


class BlockStack(nn.Module):
    def __init__(self, blockType: nn.Module, in_channels: int, out_channels: int, stride: int, dropout: float, numBlock: int):
        super(BlockStack, self).__init__() 
        self.group = self.make_group(blockType, in_channels, out_channels, stride, dropout, numBlock)
        
    def make_group(self, blockType: nn.Module, in_channels: int, out_channels: int, stride: int, dropout: float, numBlock: int):
        layers = []
        for idx in range(numBlock):
            if idx == 0:
                layers.append(blockType(in_channels, out_channels, stride, dropout)) # This changes both resolution and channels
            else:
                layers.append(blockType(out_channels, out_channels, 1, dropout))
        
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.group(x)

        

class WRN(nn.Module):
    def __init__(self, blockType: nn.Module, depth: int, widenFact: int, numClasses: int, patchSz: int = 0, dropout: float = 0.0):
        super(WRN, self).__init__()
        assert (depth - 4) % 6 == 0
        numBlock = (depth - 4) // 6

        channelDepth = [16, 16 * widenFact, 32 * widenFact, 64 * widenFact]
        strides = [1, 2, 2]

        self.patchSz = patchSz
        if patchSz != 0:
            self.stem = nn.Conv2d(3, channelDepth[0], kernel_size = patchSz, stride = patchSz) # The actual patchify layer
        else:
            self.stem = nn.Conv2d(3, channelDepth[0], kernel_size = 3, stride = 1) # Almost similar to patchify, but its overlapping kernel => no
        
        self.largeGroup = nn.ModuleList(
            [BlockStack(blockType, channelDepth[i], channelDepth[i + 1], strides[i], dropout, numBlock) for i in range(3)]
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
        if self.patchSz != 0:
            assert x.shape[0] % self.patchSz == 0, f"Patch size is enabled but is not divisible by input shape. Redo the model"
        
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
        
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x