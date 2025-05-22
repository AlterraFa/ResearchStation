from utils.transformer import *
from torchvision.models import ResNet50_Weights, resnet50


class ViT(nn.Module):
    def __init__(self, 
                 numClasses: int,
                 imgSize   : int,
                 patchSize : int,
                 imgDim    : int = 3,
                 modelDepth: int = 512,
                 nEncoder  : int = 1,
                 droprate  : float = 0.1):
        super().__init__()
        
        self.patchify = PatchEncoding(imgSize, patchSize, imgDim, modelDepth = modelDepth)
        numPatches = self.patchify.numPatches
        
        self.clsToken = nn.Parameter(data = torch.zeros(1, 1, modelDepth), requires_grad = True) # Acts as sumarization vector
        nn.init.trunc_normal_(self.clsToken, std = 0.02)

        self.position = SpatialEncoding(imgSize = (int(numPatches ** .5), int(numPatches ** .5)), modelDepth = modelDepth)

        self.encoders = nn.ModuleList([
            TransformerEncoder(dropout = droprate, ffDim = 1024) for _ in range(nEncoder)
        ])
        
        self.norm = nn.LayerNorm(modelDepth)
        self.head = nn.Linear(modelDepth, numClasses)
        
    def forward(self, x: torch.Tensor):
        x = self.patchify(x)
        # clsToken = self.clsToken.expand(x.shape[0], -1, -1)
        # x = torch.cat([clsToken, x], dim = 1)
        x = self.position(x)
        
        for encoder in self.encoders:
            x = encoder(x)
        
        x = x.mean(dim = 1)
        x = self.norm(x)
            
        return self.head(x)
    
    
    def summary(self):
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_MB = total_params * 4 / (1024 ** 2)  # Assuming 32-bit float = 4 bytes
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"Approximate Model Size: {total_MB:.2f} MB")
        
        
class DeTr(nn.Module):
    def __init__(self, numClasses: int, nEncoders: int, nDecoders: int, nHeads: int = 8, hiddenDim: int = 1024):
        super().__init__()

        strideReplace = [False, False, False]
        baseSz = 20
        outputSize = baseSz * (sum(strideReplace) + 1)
        
        backbone = resnet50(replace_stride_with_dilation = strideReplace,
                            weights = ResNet50_Weights.DEFAULT)
        
        self.proposalSize = 100
        self.backbone    = nn.Sequential(*list(backbone.children())[:-2])
        self.conv        = nn.Conv2d(2048, hiddenDim, 1)
        self.transformer = Transformer(nEncoders = nEncoders, nDecoders = nDecoders, querySz = self.proposalSize, nHeads = nHeads, 
                                    positionEnc = SpatialEncoding(imgSize = (outputSize, outputSize), modelDepth = hiddenDim), modelDepth = hiddenDim)
        
        self.linearCls = self.MLP(inputDim = hiddenDim, hiddenDim = hiddenDim * 2, outputDim = numClasses + 1)
        self.linearBB  = self.MLP(inputDim = hiddenDim, hiddenDim = hiddenDim * 2, outputDim = 4)
        
        self.hiddenDim = hiddenDim
        self.numClasses = numClasses
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x: Tensor, auxiliary = False):
        
        out = self.backbone(x)
        out = self.conv(out)
        
        B = out.shape[0]
        out = out.reshape(B, -1, self.hiddenDim).contiguous()
        out = self.transformer(out, auxiliary)
        if auxiliary:
            out = torch.stack(out, dim = 1)
        
        
        return self.linearCls(out), self.linearBB(out).sigmoid() # bounding box coordinates is formatted as (x, y, w, h) normalized to [0, 1]

    def MLP(self, inputDim, hiddenDim, outputDim, numLayers = 3):
        layers = []
        for idx in range(numLayers):
            layers += [     nn.Linear(inputDim, hiddenDim) if idx == 0 
                       else nn.Linear(hiddenDim, outputDim) if idx == numLayers - 1 
                       else nn.Sequential(nn.Linear(hiddenDim, hiddenDim), nn.LeakyReLU(inplace = True))
                       ]
            
        return nn.Sequential(*layers)

    
    def summary(self):
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_MB = total_params * 4 / (1024 ** 2)  # Assuming 32-bit float = 4 bytes
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"Approximate Model Size: {total_MB:.2f} MB")