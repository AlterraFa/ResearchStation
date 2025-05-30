import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.transformer import TransformerEncoder, NoEncoding
from einops.layers.torch import Rearrange

class AttentionPooling(nn.Module):
    def __init__(self, modelDim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(modelDim, modelDim // 2),
            nn.Tanh(),
            nn.Linear(modelDim // 2, 1)
        )

    def forward(self, x):
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled

class TabTransformer(nn.Module):
    def __init__(self, numCont: int, numCat: list[int], numClasses: int,  numLayers: int = 6, modelDim: int = 512, numHeads: int = 8, dropout: float = 0.1):
        super(TabTransformer, self).__init__()

        self.embed = nn.ModuleList([
            nn.Embedding(numCat[i] + 2, modelDim) for i in range(len(numCat))
        ])
        self.encoder = nn.ModuleList([
            TransformerEncoder(modelDim, numHead=numHeads, dropout=dropout) for _ in range(numLayers)
        ])
        
        self.batchNorm = nn.BatchNorm1d(numCont)
        self.contExpand = nn.Linear(1, modelDim)
        self.mlp = nn.Sequential(*[
            AttentionPooling(modelDim),
            nn.Linear(modelDim, modelDim // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(modelDim // 4, numClasses)
        ])
        
    def forward(self, xCont: torch.Tensor, xCat: torch.Tensor):
        
        embedOut = []
        for idx, embedding in enumerate(self.embed):
            embedOut += [embedding(xCat[:, idx]).unsqueeze(1)]
        catOutput = torch.cat(embedOut, dim = 1)

        for encoder in self.encoder:
            catOutput = encoder(catOutput, catOutput)

        contOutput = self.batchNorm(xCont).unsqueeze(-1)
        contOutput = self.contExpand(contOutput)

        output = torch.cat([catOutput, contOutput], dim = 1)
        output = self.mlp(output)

        return output
    
    def forwardPretrain(self):
        ...


class TwinTab(nn.Module):
    def __init__(self, numCont: int, numCat: list[int], numClasses: int, numEnc: int = 6, modelDim: int = 512, numHeads: int = 8, dropout: float = 0.1):
        super(TwinTab, self).__init__()

        self.embed = nn.ModuleList([
            nn.Embedding(numCat[i] + 2, modelDim) for i in range(len(numCat))
        ])
        self.encoderCont = nn.ModuleList([
            TransformerEncoder(modelDim, numHead = numHeads, dropout = dropout) for _ in range(numEnc)
        ])
        self.encoderCat = nn.ModuleList([
            TransformerEncoder(modelDim, numHead = numHeads, dropout = dropout) for _ in range(numEnc)
        ])
        self.encoderMix = TransformerEncoder(modelDim, numHead = numHeads, dropout = dropout)
        

        self.batchNorm = nn.BatchNorm1d(numCont)
        self.contExpand = nn.Linear(1, modelDim)
        self.mlp = nn.Sequential(*[
            AttentionPooling(modelDim),
            nn.Linear(modelDim, modelDim // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(modelDim // 4, numClasses)
        ])
        
    def forward(self, xCont: torch.Tensor, xCat: torch.Tensor):

        embedOut = []
        for idx, embedding in enumerate(self.embed):
            embedOut += [embedding(xCat[:, idx]).unsqueeze(1)]
        catOutput = torch.cat(embedOut, dim = 1)

        for encoder in self.encoderCat:
            catOutput = encoder(catOutput, catOutput)

        contOutput = self.batchNorm(xCont).unsqueeze(-1)
        contOutput = self.contExpand(contOutput)
        for encoder in self.encoderCont:
            contOutput = encoder(contOutput, contOutput)

        output = torch.cat([catOutput, contOutput], dim = 1)
        output = self.encoderMix(output, output)
        output = self.mlp(output)

        return output
        
    def forwardPretrain(self):
        ...