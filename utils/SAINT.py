import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.transformer import TransformerEncoder, AttentionPooling
from einops.layers.torch import Rearrange

from typing import overload, Literal, Tuple, List


class SAINTblock(nn.Module):
    def __init__(self, modelDepth: int = 512, numHeads: int = 8, ffDim: int = 2048, dropout: float = 0.1):
        super(SAINTblock, self).__init__()
        
        self.selfAtten   = TransformerEncoder(modelDepth = modelDepth, numHead = numHeads, ffDim = ffDim, dropout = dropout)
        self.transpose   = Rearrange("B F D -> F B D")
        self.intraAtten  = TransformerEncoder(modelDepth = modelDepth, numHead = numHeads, ffDim = ffDim, dropout = dropout)
        self.retranspose = Rearrange("F B D -> B F D")

    def forward(self, x: torch.Tensor):

        noEnc   = torch.zeros_like(x, requires_grad = False)
        x = self.selfAtten(x, noEnc)

        x = self.transpose(x)
        noEnc = self.transpose(noEnc)
        x = self.intraAtten(x, noEnc)

        return self.retranspose(x)
class SAINTDownstream(nn.Module):
    def __init__(self, modelDepth: int, numClasses: int, dropout = 0.1):
        super().__init__()
    
        self.downstream = nn.Sequential(*[
            AttentionPooling(modelDepth),
            nn.Linear(modelDepth, modelDepth // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(modelDepth // 4, numClasses)
        ])

    def forward(self, x: torch.Tensor):
        return self.downstream(x)
    
class SAINTPretrain(nn.Module):
    def __init__(self, numCont: int, catCount: list[int], modelDepth: int):
        super().__init__()

        self.catSize = len(catCount); self.contSize = numCont
        
        self.contrastHead = nn.Sequential(
            nn.LazyLinear(modelDepth),
            nn.LeakyReLU(inplace = True),
            nn.LazyLinear(128)
        )

        contHeads = []
        for _ in range(numCont):
            contHeads += [nn.Sequential(
                nn.LazyLinear(128),
                nn.LeakyReLU(),
                nn.LazyLinear(1), # This needs an MSE or Smooth L1 loss
            )]
        self.contHeads = nn.ModuleList(contHeads)
            
        catHeads  = []
        for lastLayer in catCount:
            catHeads += [nn.Sequential(
                nn.LazyLinear(128),
                nn.LeakyReLU(),
                nn.LazyLinear(lastLayer), # This needs a CrossEntropy or a KL divergence loss
            )]
        self.catHeads = nn.ModuleList(catHeads)
    
    def forward(self, x: torch.Tensor):
        contrastOutput = self.contrastHead(x)

        denoiseCatOutput = [
            head(x[:, idx + self.contSize, :]) for idx, head in enumerate(self.catHeads)
        ]
        denoiseContOutput = [
            head(x[:, idx , :]) for idx, head in enumerate(self.contHeads)
        ]
        
        return contrastOutput, denoiseContOutput, denoiseCatOutput
    
class SAINT(nn.Module):
    def __init__(self, 
                 numCont: int,
                 catCount: list[int],
                 numClasses: int, 
                 numStages: int,
                 modelDepth: int = 512, 
                 numHeads: int = 8, 
                 ffDim: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()

        if len(catCount):
            self.catEmbedding = nn.ModuleList([
                nn.Embedding(catCount[i] + 2, modelDepth) for i in range(len(catCount))
            ])
        
        self.clsEmbedding = nn.Parameter(torch.randn((1, 1, modelDepth)), requires_grad = True)
        
        if numCont != 0:
            self.contNorm      = nn.LayerNorm(numCont) # this might not work for inference both views at once
            self.contEmbedding = nn.Sequential(
                Rearrange("B F -> B F 1"),
                nn.Linear(1, modelDepth)
            )
            
        self.catSize = len(catCount); self.contSize = numCont
        
        self.stages = nn.Sequential(*[
            SAINTblock(modelDepth = modelDepth, numHeads = numHeads, ffDim = ffDim, dropout = dropout) 
            for _ in range(numStages)
        ])
        
        self.downstream = SAINTDownstream(modelDepth = modelDepth, numClasses = numClasses, dropout = 0.1)
        self.pretrain   = SAINTPretrain(numCont = numCont, catCount = catCount, modelDepth = modelDepth)


    @overload
    def forward(self, xCont: torch.Tensor, xCat: torch.Tensor, *, pretrainMode: Literal[False]) -> torch.Tensor: ...
    
    @overload
    def forward(self, xCont: None, xCat: None, *, embedding: torch.Tensor, pretrainMode: Literal[True]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: ...

    def forward(self, 
                xCont: torch.Tensor = None, 
                xCat: torch.Tensor = None, 
                embedding: torch.Tensor = None, 
                pretrainMode: bool = False):

        assert not ((xCont is not None or xCat is not None) and embedding is not None), "Both Embeddings and Raw data input is not allowed"

        if pretrainMode:
            assert embedding is not None, "Pretrain mode enabled. Embedding from `self.getEmbedding` is required"
        else:
            assert xCont is not None or xCat is not None, "Pretrain mode disabled. Continuous and categorical data is required"
            embedding = self.get_embedding(xCont, xCat, pretrainMode = pretrainMode)

        # Core
        output = self.stages(embedding)
        
        # Head
        if pretrainMode == False:
            output = self.downstream(output)
            return output
        else:
            contrastOutput, denoiseContOutput, denoiseCatOutput = self.pretrain(output)
            return contrastOutput, denoiseContOutput, denoiseCatOutput
        
    def get_embedding(self, xCont: torch.Tensor, xCat: torch.Tensor, pretrainMode = False):
        # Embedding (Tail)
        contNorm = self.contNorm(xCont)
        contEmbed = self.contEmbedding(contNorm)
        
        catEmbed = []
        for idx, e in enumerate(self.catEmbedding):
            catEmbed += [e(xCat[:, idx])]
        catEmbed = torch.stack(catEmbed, dim = 1)
        
        if pretrainMode == False:
            B, _= xCat.shape
            clsToken = self.clsEmbedding.tile(B, 1, 1)
            embedding = torch.cat([clsToken, contEmbed, catEmbed], dim = 1)
        else:
            embedding = torch.cat([contEmbed, catEmbed], dim = 1)
            
        return embedding 