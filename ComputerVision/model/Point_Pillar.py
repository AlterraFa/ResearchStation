import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

import torch
import torch.nn as nn

from utils.detectionHead import SSD
from utils.WRN import ResnetBlock


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0):
        super().__init__()
        
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            output_padding = output_padding,
            bias         = False
        )
        
        self.norm = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x: torch.Tensor):
        return self.relu(self.norm(self.transpose_conv(x)))

class PillarBackbone(nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        
        pyramid = []; deconvs = []; transpose_arg = [[1, 1, 0], [4, 2, 1], [8, 4, 2]]
        for reduce_iter in range(3):
            conv = ResnetBlock(target_dim * (2 ** reduce_iter),
                               target_dim * (2 ** (reduce_iter + 1)),
                               stride = 2, dropout = 0.01)
            transpose = DeconvBlock(
                in_channels  = target_dim * (2 ** (reduce_iter + 1)),
                out_channels = target_dim * (2 ** 2),
                kernel_size  = transpose_arg[reduce_iter][0],
                stride       = transpose_arg[reduce_iter][1],
                padding      = transpose_arg[reduce_iter][2],
                output_padding = 0,
            )
            
            
            pyramid += [conv]
            deconvs += [transpose]

        self.pyramid = nn.ModuleList(pyramid)
        self.deconv  = nn.ModuleList(deconvs)
        
    def forward(self, out: torch.Tensor):
        out_scale = []
        for downsample, upsample in zip(self.pyramid, self.deconv):
            out = downsample(out)
            out_scale += [upsample(out)]
            
        return torch.concat(out_scale, dim = 1)
        
        

class PointPillar(nn.Module):
    def __init__(self,
                 num_classes: int, 
                 loc_features: int, 
                 point_dim: int, 
                 target_dim: int,
                 height: int,
                 width: int):
        super().__init__()

        self.H = height
        self.W = width

        self.pointcloud_extract = nn.Sequential(*[
            nn.Conv2d(point_dim, target_dim, kernel_size = (1, 1)),
            nn.BatchNorm2d(target_dim), 
            nn.LeakyReLU(negative_slope = 0.05, inplace = True)
        ])

        self.backbone  = PillarBackbone(target_dim)
        self.detection = SSD(num_classes = num_classes, 
                             loc_features = loc_features, 
                             scale = [target_dim * 4, target_dim, target_dim // 2], 
                             num_boxes = [3, 4, 3, 3], 
                             input_dim = target_dim * 4 * 3)
        
    def forward(self, pointcloud: torch.Tensor, pillar_index: torch.Tensor):

        out = self.pointcloud_extract(pointcloud)
        out = out.amax(dim = -1)
        
        
        out = self.warp_to_grid(out, pillar_index)        
        out = self.backbone(out)
        
        return self.detection(out)

    def warp_to_grid(self, a: torch.Tensor, pillar_index: torch.Tensor):
        B, C, P = a.shape
        HW      = self.H * self.W 
        
        assert pillar_index.min() >= 0
        assert pillar_index.max() != HW
        
        feat_canvas  = a.new_zeros(B, C, HW)
        
        scatter_idx  = pillar_index.unsqueeze(1).expand(-1, C, -1)
        
        feat_canvas.scatter_(dim = 2, index = scatter_idx, src = a)
        
        return feat_canvas.reshape(B, C, self.H, self.W)