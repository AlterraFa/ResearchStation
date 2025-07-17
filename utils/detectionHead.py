import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn

class SSD(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 loc_features: int, 
                 scale: list[int],
                 num_boxes: list[int], 
                 input_dim: int = 512):

        super(SSD, self).__init__()
        self.num_classes  = num_classes
        self.loc_features = loc_features

        assert len(scale) != 0, "A scale specification is needed"
        assert len(num_boxes) != 0, "A num of boxes specification is needed"
        assert len(scale) + 1 == len(num_boxes), "List of boxes not equal to list of scale + 1"


        # Additional layers for SSD
        extra = []
        for idx in range(len(scale)):
            if idx == 0:
                extra += [
                    nn.Sequential(
                    nn.Conv2d(input_dim, scale[idx], kernel_size = 3, padding = 1, dilation = 1),
                    nn.ReLU(inplace = True)
                )]
            else:
                try: # out of index check
                    bottleneck_size = scale[idx - 1] // 2 if scale[idx - 1] // 2 < scale[idx] else scale[idx] // 2
                    extra += [
                        nn.Sequential(
                            nn.Conv2d(scale[idx - 1], bottleneck_size, kernel_size = 1),
                            nn.ReLU(inplace = True),
                            nn.Conv2d(bottleneck_size, scale[idx], kernel_size = 3, stride = 2, padding = 1),
                            nn.ReLU(inplace = True)
                        )]
                except: break
        self.extras = nn.ModuleList(extra)

        # Localization and class prediction layers
        loc = []; conf = []
        for idx in range(len(num_boxes)):
            if idx == 0:
                loc  += [nn.Conv2d(input_dim, num_boxes[idx] * self.loc_features, kernel_size = 3, padding = 1)]
                conf += [nn.Conv2d(input_dim, num_boxes[idx] * num_classes, kernel_size = 3, padding = 1)]
            else:
                loc  += [nn.Conv2d(scale[idx - 1], num_boxes[idx] * self.loc_features, kernel_size = 3, padding = 1)]
                conf += [nn.Conv2d(scale[idx - 1], num_boxes[idx] * num_classes, kernel_size = 3, padding = 1)]

        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

    def forward(self, x):
        locs = []
        confs = []

        locs.append(self.loc[0](x).permute(0, 2, 3, 1).contiguous())
        confs.append(self.conf[0](x).permute(0, 2, 3, 1).contiguous())


        for (i, layer) in enumerate(self.extras):
            x = layer(x)
            locs.append(self.loc[i+1](x).permute(0, 2, 3, 1).contiguous())
            confs.append(self.conf[i+1](x).permute(0, 2, 3, 1).contiguous())

        locs = torch.cat([o.view(o.size(0), -1) for o in locs], 1)
        confs = torch.cat([o.view(o.size(0), -1) for o in confs], 1)

        locs = locs.view(locs.size(0), -1, self.loc_features)
        confs = confs.view(confs.size(0), -1, self.num_classes)

        return locs, confs
    
if __name__ == "__main__":
    num_classes  = 21
    loc_features = 4
    num_boxes    = [4, 6, 6, 6, 4 ,4]
    scale        = [1024, 512, 256, 256, 256]
    
    ssd = SSD(num_classes, 
              loc_features, 
              num_boxes = num_boxes,
              scale = scale)
    x = torch.randn(1, 512, 18, 18)
    locs, confs = ssd(x)
    
    print("Localization predictions:", locs.size())
    print("Confidence predictions:", confs.size())