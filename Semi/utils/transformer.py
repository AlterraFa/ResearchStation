import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
    
import torch
from torch import nn, Tensor
from transformers import GPT2Tokenizer
from typing import Union, Tuple, Optional



class VocabEncoding:
    """
    A module for encoding vocabulary into a tensor representation. This module takes a list of 
    words or sentences and converts them into a tensor format that can be used as input for neural 
    networks. The encoding is based on a predefined vocabulary, and each word is represented by a unique index.

    Args:
        vocabLimit (int, optional): The maximum number of vocabulary words to use. If None, use all available words.
        maxToken (int, optional): The maximum number of tokens in a sequence. Default is 500.
        dtype (torch.dtype, optional): The data type of the tensor. Default is torch.float32.
        device (torch.device, optional): The device on which to place the tensor. Default is CPU.

    Attributes:
        device (torch.device): The device on which to place the tensor.
        vocabLimit (int): The maximum number of vocabulary words to use.
        maxToken (int): The maximum number of tokens in a sequence.
        dtype (torch.dtype): The data type of the tensor.
        wordIdx (dict): A dictionary mapping words to their indices in the vocabulary.
    """
    def __init__(self, 
                 maxToken = 500,
                 dtype = torch.float32,
                 device = torch.device('cpu')):
        self.device = device
        self.maxToken = maxToken
        self.dtype = dtype
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocabSize = self.tokenizer.vocab_size + 1
        
    def encode(self, x: tuple[Union[list, str], ...]):
        encodedInputs = self.tokenizer(x, padding = 'max_length', truncation = True, max_length = self.maxToken, return_tensors = 'pt')
        
        indices = encodedInputs['input_ids'].to(self.device)
        encodedData = torch.zeros(*(len(x), self.maxToken, self.vocabSize),
                                  dtype = self.dtype,
                                  device = self.device)
        
        encodedData.scatter_(2, indices.unsqueeze(-1), 1.0)
        
        return encodedData
       
class PatchEncoding(nn.Module):
    def __init__(self, imgSize: int, patchSize: int, imgDim: int = 3, modelDepth: int = 512):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.module.{self.__class__.__name__}")
        assert imgSize % patchSize == 0
        
        self.proj = nn.Conv2d(in_channels  = imgDim,
                              out_channels = modelDepth,
                              kernel_size  = patchSize,
                              stride       = patchSize)
        
        self.numPatches = (imgSize // patchSize) ** 2
        self.norm = nn.LayerNorm(modelDepth)
        
    def forward(self, x: torch.Tensor):
        x = self.proj(x) # (B, D, H // pSize, W // pSize) if square
        x = x.flatten(2) # (B, D, numPatch)
        x = x.transpose(2, 1) # (B, numPatch, D)
        return self.norm(x)
         
 
class PositionalEncoding(nn.Module):
    """
    A module for adding positional encoding to the input tensor. Positional encoding is used to 
    provide information about the position of each token in the sequence, which is important for 
    sequence-based models like transformers. This module adds sinusoidal positional encodings to the input tensor.

    Args
        inputSize (torch.Size): The size of the input tensor.
        modelDepth (int, optional): The depth of the model. Default is 512.

    Attributes
        modelDepth (int): The depth of the model.
        embed (nn.Linear): A linear layer for embedding the input tensor.
        position (torch.Tensor): A tensor representing the positions in the sequence.
        depth (torch.Tensor): A tensor representing the depth of the model.
    """
    def __init__(self, 
                 inputSize: int, 
                 modelDepth: int = 512,
                 ) -> None:

        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.module.{self.__class__.__name__}")

        position = torch.arange(inputSize).unsqueeze(1)
        depth = torch.arange(modelDepth).unsqueeze(0)
        
        self.register_buffer("PE", torch.where(depth % 2 == 0, 
                                               torch.sin(position / torch.pow(10000, depth / modelDepth)), 
                                               torch.cos(position / torch.pow(10000, depth / modelDepth))))
    
    
    def forward(self, x: Tensor):
        return self.PE
    
class SpatialEncoding(nn.Module):
    def __init__(self, imgSize: Tuple[int, int], modelDepth: int = 512):
        super().__init__()
        assert len(imgSize) == 2
        
        H, W = imgSize
        D = modelDepth
        
        u = torch.arange(0, H, 1).view(1, -1, 1)
        v = torch.arange(0, W, 1).view(1, 1, -1)
        dimension = torch.arange(0, D, 1).view(-1, 1, 1)
        remainder = (dimension % 4).squeeze()
        
        denom = torch.pow(1e4, (2 * (dimension // 4) / D))
        
        PE2D = torch.zeros([D, H, W])
        PE2D[remainder == 0, ...] = torch.sin(u / denom[remainder == 0, ...])
        PE2D[remainder == 1, ...] = torch.cos(u / denom[remainder == 1, ...])
        PE2D[remainder == 2, ...] = torch.sin(v / denom[remainder == 2, ...])
        PE2D[remainder == 3, ...] = torch.cos(v / denom[remainder == 3, ...])

        self.register_buffer("PE", PE2D.flatten(1).T)
        
    def forward(self, x: Tensor):
        
        return self.PE        
    
class NoEncoding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor):
        return torch.zeros_like(x, device = x.device)

class MultiheadAttention(nn.Module):
    def __init__(self, 
                 modelDepth: int = 512,
                 numHeads: int = 8): 
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.module.{self.__class__.__name__}")
        
        self.numHeads = numHeads

        self.qTransform = nn.Linear(modelDepth, modelDepth)
        self.kTransform = nn.Linear(modelDepth, modelDepth)
        self.vTransform = nn.Linear(modelDepth, modelDepth)
        self.reverseTransform = nn.Linear(modelDepth, modelDepth)


    def forward(self, 
                x: Tuple[Tensor, Tensor, Tensor] | Union[Tensor, Tensor, Tensor], 
                useMask: bool = False):
        
        if not torch.jit.is_tracing():
            assert x[0].shape[-1] % self.numHeads == 0, f"Number of heads must be divisible by the a word depth, got {x[0].shape[-1]} % {self.numHeads}"


        q = self.qTransform(x[0])
        k = self.kTransform(x[1])
        v = self.vTransform(x[2])
        
        q = q.reshape(*q.shape[:2], self.numHeads, -1).permute(0, 2, 1, 3) # (B, Head, Token, Dim)
        k = k.reshape(*k.shape[:2], self.numHeads, -1).permute(0, 2, 1, 3)
        v = v.reshape(*v.shape[:2], self.numHeads, -1).permute(0, 2, 1, 3)

        cosineSimilarities = q @ k.permute(0, 1, 3, 2) / (k.shape[-1] ** .5)
        if useMask:
            inf = torch.full([cosineSimilarities.shape[-1]] * 2, - torch.inf)
            mask = torch.triu(inf, diagonal = 1)
            mask.to(device = cosineSimilarities.device)
            cosineSimilarities += mask
    
        probability = torch.softmax(cosineSimilarities, -1)
        attentionScore = (probability @ v).permute(0, 2, 1, 3)
        concat = attentionScore.flatten(start_dim = 2)
        reproject = self.reverseTransform(concat)

        return reproject
    
class FeedForward(nn.Module):
    def __init__(self, 
                 dropout: float = .1, 
                 modelDepth: int = 512,
                 ffDim: int = 2048):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.module.{self.__class__.__name__}")
        
        self.expansion = nn.Linear(modelDepth, ffDim)
        self.shrinkage = nn.Linear(ffDim, modelDepth)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor):
        # max(0, W1x+b)W2 + b
        expand = self.dropout(self.relu(self.expansion(x)))
        shrink = self.dropout(self.shrinkage(expand))
        
        
        return shrink
    
class LayerNormalization(nn.Module):
    def __init__(self, modelDepth: int = 512) -> None:
        super().__init__()
        # CORRECT: γ starts at 1, β at 0 for identity init
        self.gamma = nn.Parameter(torch.ones(modelDepth))
        self.beta  = nn.Parameter(torch.zeros(modelDepth))
    def forward(self, x):
        μ  = x.mean(dim=-1, keepdim=True)
        σ2 = (x - μ).pow(2).mean(dim=-1, keepdim=True)
        σ  = torch.sqrt(σ2 + 1e-5)
        return self.gamma * ((x - μ)/σ) + self.beta  
        

class TransformerEncoder(nn.Module):
    def __init__(self,
                 modelDepth: int = 512,
                 numHead: int = 8,
                 ffDim = 2048,
                 dropout = .1):
        super(TransformerEncoder, self).__init__()
        torch._C._log_api_usage_once(f"torch.nn.module.{self.__class__.__name__}")
        

        self.MABlock = MultiheadAttention(modelDepth = modelDepth, 
                                          numHeads = numHead)
        
        self.FFBlock = FeedForward(modelDepth = modelDepth,
                                   dropout = dropout,
                                   ffDim = ffDim)
        
        self.LNBlock1 = LayerNormalization(modelDepth)
        self.LNBlock2 = LayerNormalization(modelDepth)

    def forward(self, x: Tensor, xPos: Tensor):

        q = k = xPos + x
        MAout = self.MABlock([q, k, x])
        MAout = self.LNBlock1(MAout + x)
        
        output = self.FFBlock(MAout)
        output = self.LNBlock2(output + MAout)

        return output
        
class TransformerDecoder(nn.Module):
    def __init__(self,
                 modelDepth: int = 512,
                 numHeads: int = 8,
                 ffDim = 2048,
                 dropout = .1) -> None:
        super(TransformerDecoder, self).__init__()
        torch._C._log_api_usage_once(f"torch.nn.module.{self.__class__.__name__}")
 
        self.selfAtten = MultiheadAttention(modelDepth = modelDepth,
                                           numHeads = numHeads)
                

        self.crossAtten = MultiheadAttention(modelDepth = modelDepth, 
                                          numHeads = numHeads)
        

        self.FFBlock = FeedForward(modelDepth = modelDepth,
                                   dropout = dropout,
                                   ffDim = ffDim)

        self.LNBlock1 = LayerNormalization(modelDepth)
        self.LNBlock2 = LayerNormalization(modelDepth)
        self.LNBlock3 = LayerNormalization(modelDepth)


    def forward(self, tgt: Tensor, tgtPos: Tensor, mem: Tensor, memPos: Tensor):
        
        q = k = tgtPos + tgt
        out = self.selfAtten([q, k, tgt], False)
        tgt = self.LNBlock1(out + tgt)

        k = memPos + mem; q = tgt + tgtPos
        out = self.crossAtten([q, k, mem])
        tgt = self.LNBlock2(out + tgt)
        
        output = self.FFBlock(tgt)
        tgt = self.LNBlock3(tgt + output)
        
        return tgt

class Transformer(nn.Module):
    def __init__(self, nEncoders: int, nDecoders:int, querySz: int, nHeads = 8, ffnDim: int = 1024, modelDepth: int = 512, positionEnc: Optional[nn.Module] = None, droprate: float = 0):
        super().__init__()
        
        self.encoders = nn.ModuleList([TransformerEncoder(modelDepth = modelDepth, ffDim = ffnDim, dropout = droprate, numHead = nHeads) for _ in range(nEncoders)])
        self.decoders = nn.ModuleList([TransformerDecoder(modelDepth = modelDepth, ffDim = ffnDim, dropout = droprate, numHeads = nHeads) for _ in range(nDecoders)])
        self.positionEnc = positionEnc or NoEncoding()
        
        self.tgtPos = nn.Embedding(querySz, modelDepth)

        
    def forward(self, src: torch.Tensor, auxiliary: bool = False):
        
        srcPos = self.positionEnc(src) # This thing returns the Positional Encoding tensors
        
        for encoder in self.encoders:
            src = encoder(src, srcPos)
        
        B = src.shape[0]
        tgtPos = self.tgtPos.weight
        tgtPos = tgtPos.expand(B, -1, -1)
        tgt = torch.zeros_like(tgtPos, device = src.device)

        auxLogits = []
        for idx, decoder in enumerate(self.decoders):
            tgt = decoder(tgt, tgtPos, src, srcPos)
            if auxiliary and self.training: auxLogits += [tgt]
        
        return tgt if not (auxiliary and self.training) else auxLogits

    def summary(self):
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_MB = total_params * 4 / (1024 ** 2)  # Assuming 32-bit float = 4 bytes
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"Approximate Model Size: {total_MB:.2f} MB")
        


class Pooling(nn.Module):
    def __init__(self, dim):
        super(Pooling, self).__init__()
        self.pooling = nn.AdaptiveMaxPool1d(dim)
        
    def forward(self, x: Tensor):
        return self.pooling(x.permute(0, 2, 1)).squeeze(2)