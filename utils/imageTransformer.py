from utils.transformer import *
from torchvision.models import ResNet50_Weights, resnet50
import torch.nn.functional as F

from torchvision.ops import generalized_box_iou as GIoU, box_convert as bboxConvert
from scipy.optimize import linear_sum_assignment as hungarianMatch

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
        self.nEncoders = nEncoders
        self.nDecoders = nDecoders
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
        layers += [nn.LayerNorm(inputDim)]
        for idx in range(numLayers):
            layers += [     nn.Linear(inputDim, hiddenDim) if idx == 0 
                       else nn.Linear(hiddenDim, outputDim) if idx == numLayers - 1 
                       else nn.Sequential(nn.Linear(hiddenDim, hiddenDim), nn.LeakyReLU(inplace = True))
                       ]
            
        return nn.Sequential(*layers)
    
    def hungarianLoss(self, 
                      classGT: Tensor, classLogits: Tensor, 
                      bboxGT: Tensor, bboxProposal: Tensor,
                      classCriterion: nn.Module, boxCriterion: nn.Module,
                      alphaClass: float = 1.0, alphaL1Box: float = 1.0, alphaGIoUBox: float = 1.0):
        numDetections = classGT.shape[0]
        
        # Holy fuck, a lot of dimension manipulation 
        # This is horseshit
        labelsFlat        = classGT.unsqueeze(0).expand(self.proposalSize, -1).reshape(-1)
        classLogitsFlat   = classLogits.unsqueeze(1).expand(-1, numDetections, -1).reshape(-1, self.numClasses + 1)
        hungarianClassMat = classCriterion(classLogitsFlat, labelsFlat).reshape(self.proposalSize, -1)

        bboxPredExpandedFlat = bboxProposal.unsqueeze(1).expand(-1, numDetections, -1).reshape(-1, 4)
        bboxGTExpandedFlat   = bboxGT.unsqueeze(0).expand(self.proposalSize, -1, -1).reshape(-1, 4)
        bboxL1Mat            = boxCriterion(bboxPredExpandedFlat, bboxGTExpandedFlat).reshape(self.proposalSize, numDetections, -1).mean(dim = -1)
        bboxGIoUMat          = GIoU(bboxProposal, bboxGT)

        hungarianCost = alphaClass * hungarianClassMat + alphaL1Box * bboxL1Mat + alphaGIoUBox * - bboxGIoUMat
        hungarianCost = self.sanitizeCost(hungarianCost)
        hungarianCost = hungarianCost.cpu().detach().numpy()

        try:
            rowIdx, colIdx = hungarianMatch(hungarianCost)
        except Exception as e:
            print("Hungarian Error: ", e)
            print("Hungarian Cost: Inf - ", torch.any(torch.tensor(hungarianCost).isinf()), "NaN - ", torch.any(torch.tensor(hungarianCost).isnan()))
            print("Hungarian Class Mat: Inf - ", torch.any(hungarianClassMat.isinf()), "NaN - ", torch.any(hungarianClassMat.isnan()))
            print("Hungarian BBox L1 Mat: Inf - ", torch.any(bboxL1Mat.isinf()), "NaN - ", torch.any(bboxL1Mat.isnan()))
            print("Hungarian BBox GIoU Mat: Inf - ", torch.any(bboxGIoUMat.isinf()), "NaN - ", torch.any(bboxGIoUMat.isnan()))
            
            x1p,y1p,x2p,y2p = bboxProposal.unbind(-1)
            x1g,y1g,x2g,y2g = bboxGT      .unbind(-1)

            bad_p = ((x2p <= x1p) | (y2p <= y1p)).nonzero()
            bad_g = ((x2g <= x1g) | (y2g <= y1g)).nonzero()
            print("degenerate proposals at", bad_p, "degenerate GTs at", bad_g)
            
            exit(-5)
        
        return rowIdx, colIdx, hungarianClassMat, bboxL1Mat, bboxGIoUMat
    
    def loss(self, classLogits: Tensor, labels: List[Tensor],
             bboxProposal: Tensor, bbox: List[Tensor], 
             classCriterion: nn.Module, boxCriterion: nn.Module,
             alphaClass: float = 1.0, alphaL1Box: float = 1.0, alphaGIoUBox: float = 1.0):

             
        device = classLogits.device
        batchSize = classLogits.shape[0]
        if len(classLogits.shape) == 4:
            H = classLogits.shape[1]
        else:
            H = 1

        # Since the hungarian algorithm works with 2D tensors, we need to loop through the batch
        # Or I need to write custom hungarian algorithm for 3D tensors
        
        loss = 0.0
        for i in range(batchSize):
            numDetections = labels[i].shape[0]
            classGT = labels[i].to(device)
            bboxGT  = bbox[i].to(device) 

            for headIdx in range(H):
                bboxConverted = bboxConvert(bboxProposal[i][headIdx] if H > 1 else bboxProposal[i],
                                            in_fmt = "cxcywh", out_fmt = "xyxy")
                bboxConverted = self.sanitizeBox(bboxConverted)
                singleClassLogits = classLogits[i][headIdx] if H > 1 else classLogits[i]

                if numDetections == 0:
                    classTargets = torch.full((self.proposalSize, ), self.numClasses, device = device, dtype = torch.long)
                    classLoss    = classCriterion(singleClassLogits, classTargets).mean()
                    loss         += alphaClass * classLoss
                    continue

                rowIdx, colIdx, hungarianClassMat, bboxL1Mat, bboxGIoUMat = self.hungarianLoss(classGT = classGT, classLogits = singleClassLogits, 
                                                                            bboxGT  = bboxGT, bboxProposal = bboxConverted,
                                                                            classCriterion = classCriterion, boxCriterion = boxCriterion, 
                                                                            alphaClass = alphaClass, alphaL1Box = alphaL1Box, alphaGIoUBox = alphaGIoUBox)
                
                classTargets = torch.full((self.proposalSize, ), self.numClasses, device = device, dtype = torch.long)
                classTargets[rowIdx] = classGT[colIdx]
                classLoss = classCriterion(singleClassLogits, classTargets).mean()

                
                bboxL1Loss = boxCriterion(bboxConverted[rowIdx], bboxGT[colIdx]).mean()
                bboxGIoULoss = (1.0 - GIoU(bboxConverted, bboxGT)[rowIdx, colIdx])
                bboxGIoULoss = self.sanitizeCost(bboxGIoULoss).mean()
        
                loss += alphaClass * classLoss + alphaL1Box * bboxL1Loss + alphaGIoUBox * bboxGIoULoss
                if loss < 0:
                    print("Loss is negative, something is wrong")
                    print("GIoU Loss: ", bboxGIoULoss)
                    print(bboxGIoULoss.max(), bboxGIoULoss.min(), bboxGIoULoss.mean())
                    print(bboxGT.max(), bboxGT.min(), bboxGT.mean())
                    exit(-5)
                    
            
        loss /= batchSize * H
        return loss
    
    @staticmethod
    def sanitizeBox(xyxy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x1, y1, x2, y2 = xyxy.unbind(-1)
        xa = torch.min(x1, x2)
        xb = torch.max(x1, x2)
        ya = torch.min(y1, y2)
        yb = torch.max(y1, y2)

        xb = torch.maximum(xb, xa + eps)
        yb = torch.maximum(yb, ya + eps)

        clean = torch.stack([xa, ya, xb, yb], dim=-1)
        return torch.nan_to_num(clean, nan=0.0, posinf=1.0, neginf=0.0)
    
    @staticmethod
    def sanitizeCost(cost: torch.Tensor, clip_val: float = 1e6) -> torch.Tensor:
        return torch.nan_to_num(cost, nan=clip_val, posinf=clip_val, neginf=clip_val)
    
    def summary(self):
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_MB = total_params * 4 / (1024 ** 2)  # Assuming 32-bit float = 4 bytes
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"Approximate Model Size: {total_MB:.2f} MB")