import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
    
import torch
from torch import nn, Tensor
from transformers import GPT2Tokenizer
from typing import Union, Tuple, List



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
        return x + self.PE
    

class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for a grid of size (H, W).
    Produces an encoding of shape [H*W, D], interleaving sin/cos on both axes.

    Args:
        height (int): number of grid rows (e.g. image_height // patch_height)
        width  (int): number of grid cols (e.g. image_width  // patch_width)
        dim    (int): embedding dimension D (must be multiple of 4)
    """
    def __init__(self, height: int, width: int, dim: int, temperature: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, "D must be multiple of 4 for 2D sin/cos"
        self.height = height
        self.width  = width
        self.dim    = dim

        # compute the 1D frequencies for sin/cos
        dim_quarter = dim // 4
        omega = torch.arange(dim_quarter, dtype=torch.float32) / (dim_quarter - 1)
        omega = 1.0 / (temperature ** omega)            # [D/4]

        # create grid of (y,x) positions
        y_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # [H,1]
        x_pos = torch.arange(width,  dtype=torch.float32).unsqueeze(1)  # [W,1]

        # compute sin/cos for each axis
        y_emb = y_pos * omega        # [H, D/4]
        x_emb = x_pos * omega        # [W, D/4]

        # apply sin/cos
        y_sin = torch.sin(y_emb)     # [H, D/4]
        y_cos = torch.cos(y_emb)     # [H, D/4]
        x_sin = torch.sin(x_emb)     # [W, D/4]
        x_cos = torch.cos(x_emb)     # [W, D/4]

        # now combine to get (H*W, D)
        # for each patch at (i,j), embedding = [x_sin[j], x_cos[j], y_sin[i], y_cos[i]]
        # expand & interleave
        # x part: repeat each of W rows for H times
        x_sin = x_sin.unsqueeze(0).repeat(height, 1, 1)  # [H, W, D/4]
        x_cos = x_cos.unsqueeze(0).repeat(height, 1, 1)
        # y part: repeat each of H rows across W cols
        y_sin = y_sin.unsqueeze(1).repeat(1, width, 1)   # [H, W, D/4]
        y_cos = y_cos.unsqueeze(1).repeat(1, width, 1)

        # concat along feature axis → [H, W, D]
        pe = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)
        # flatten to [H*W, D]
        pe = pe.view(height * width, dim)

        # register as buffer so it moves with .to(device)
        self.register_buffer('pos_emb', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [B, H*W, D]
        Returns:
            x + pos_emb: shape [B, H*W, D]
        """
        return x + self.pos_emb.unsqueeze(0)

class MultiheadAttention(nn.Module):
    """
    A module for multi-head attention with residual connections and layer normalization. 
    This module performs multi-head attention, adds a residual connection, and applies layer normalization 
    to the output. It is a key component of the transformer architecture.

    Args:
        modelDepth (int, optional): The depth of the model. Default is 512.
        numHeads (int, optional): The number of attention heads. Default is 8.

    Attributes:
        numHeads (int): The number of attention heads.
        device (torch.device): The device on which to place the tensor.
        qTransform (nn.Linear): A linear layer for transforming the query tensor.
        kTransform (nn.Linear): A linear layer for transforming the key tensor.
        vTransform (nn.Linear): A linear layer for transforming the value tensor.
        reverseTransform (nn.Linear): A linear layer for transforming the concatenated tensor.
        gamma (nn.Parameter): A learnable parameter for layer normalization.
        beta (nn.Parameter): A learnable parameter for layer normalization.
    """
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
        
        
        assert x[0].shape[-1] % self.numHeads == 0, f"Number of heads must be divisible by the a word depth, got {x[0].shape[-1]} % {self.numHeads}"

        q = self.qTransform(x[0])
        k = self.kTransform(x[1])
        v = self.vTransform(x[2])
        
        q = q.reshape(*q.shape[:2], self.numHeads, -1).permute(0, 2, 1, 3) # (B, Head, Token, Dim)
        k = k.reshape(*k.shape[:2], self.numHeads, -1).permute(0, 2, 1, 3)
        v = v.reshape(*v.shape[:2], self.numHeads, -1).permute(0, 2, 1, 3)

        cosineSimilarities = q @ k.permute(0, 1, 3, 2) / (k.shape[-1] ** .5)
        if useMask:
            inf = torch.full([cosineSimilarities.shape[-1]] * 2, - torch.inf).to(cosineSimilarities.device)
            mask = torch.triu(inf, diagonal = 1)
            mask.to(device = self.device)
            cosineSimilarities += mask
    
        probability = torch.softmax(cosineSimilarities, -1)
        attentionScore = (probability @ v).permute(0, 2, 1, 3)
        concat = attentionScore.reshape(*attentionScore.shape[:2], torch.prod(torch.tensor(attentionScore.shape[2:])))
        reproject = self.reverseTransform(concat)

        return reproject
    
class FeedForward(nn.Module):
    """
    A module for feed-forward neural network with residual connections and layer normalization. 
    This module consists of two linear layers with a ReLU activation in between, followed by dropout, 
    residual connection, and layer normalization.

    Args:
        inputSize (torch.Size): The size of the input tensor.
        dropout (float, optional): The dropout rate. Default is 0.1.
        modelDepth (int, optional): The depth of the model. Default is 512.
        ffDim (int, optional): The dimension of the feed-forward layer. Default is 2048.
        dtype (torch.dtype, optional): The data type of the tensor. Default is torch.float32.
        device (torch.device, optional): The device on which to place the tensor. Default is CPU.

    Attributes:
        expansion (nn.Linear): A linear layer for expanding the input tensor.
        shrinkage (nn.Linear): A linear layer for shrinking the expanded tensor.
        relu (nn.ReLU): A ReLU activation function.
        dropout (nn.Dropout): A dropout layer.
        gamma (nn.Parameter): A learnable parameter for layer normalization.
        beta (nn.Parameter): A learnable parameter for layer normalization.
    """
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
    """
    A module for layer normalization. This module normalizes the input tensor along the last dimension, 
    ensuring that the mean and variance of the tensor are consistent across different batches. Layer 
    normalization is used to stabilize and accelerate the training of deep neural networks.

    Args:
        inputSize (torch.Size): The size of the input tensor.
        dtype (torch.dtype, optional): The data type of the tensor. Default is torch.float32.
        device (torch.device, optional): The device on which to place the tensor. Default is CPU.

    Attributes:
        gamma (nn.Parameter): A learnable scaling parameter for layer normalization.
        beta (nn.Parameter): A learnable shifting parameter for layer normalization.
    """
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
    """
    A module for the Transformer encoder, consisting of positional encoding, multi-head attention, 
    and feed-forward layers. This module takes an input tensor, adds positional encoding, applies multi-head 
    attention, and processes the result through a feed-forward neural network. It is a key component of the transformer architecture.

    Args:
        inputSize (torch.Size): The size of the input tensor.
        modelDepth (int, optional): The depth of the model. Default is 512.
        numHead (int, optional): The number of attention heads. Default is 8.
        ffDim (int, optional): The dimension of the feed-forward layer. Default is 2048.
        dropout (float, optional): The dropout rate. Default is 0.1.

    Attributes:
        PEBlock (PositionalEncoding): The positional encoding block.
        RMABlock (ResidualMultiheadAttention): The multi-head attention block.
        RFFBlock (ResidualFeedForward): The feed-forward block.
    """
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
        
        self.LNBlock1 = nn.LayerNorm(modelDepth)
        self.LNBlock2 = nn.LayerNorm(modelDepth)
        
    def forward(self, x: torch.Tensor):

        MAout = self.MABlock([x] * 3)
        MAout = self.LNBlock1(MAout + x)
        
        output = self.FFBlock(MAout)
        output = self.LNBlock2(output + MAout)

        return output
        
class TransformerDecoder(nn.Module):
    def __init__(self,
                 modelDepth: int = 512,
                 numHeads: int = 8,
                 ffDim = 2048,
                 dropout = .1,
                 dtype = torch.float32,
                 device = torch.device('cpu')) -> None:
        super(TransformerDecoder, self).__init__()
        torch._C._log_api_usage_once(f"torch.nn.module.{self.__class__.__name__}")
 
        self.MMABlock = MultiheadAttention(modelDepth = modelDepth,
                                           numHeads = numHeads,
                                           dtype = dtype,
                                           device = device)
                

        self.MABlock = MultiheadAttention(modelDepth = modelDepth, 
                                          numHeads = numHeads, 
                                          dtype = dtype,
                                          device = device)
        

        self.FFBlock = FeedForward(dropout = dropout,
                                   ffDim = ffDim,
                                   dtype = dtype,
                                   device = device)

        self.LNBlock1 = LayerNormalization(modelDepth, dtype = dtype, device = device)
        self.LNBlock2 = LayerNormalization(modelDepth, dtype = dtype, device = device)
        self.LNBlock3 = LayerNormalization(modelDepth, dtype = dtype, device = device)

    def forward(self, x: Tensor, xEncoder: Tensor):
        
        MMAout = self.MMABlock([x] * 3, True)
        MMAout = self.LNBlock1(MMAout + x)
        
        MAout = self.MABlock([MMAout, xEncoder, xEncoder])
        MAout = self.LNBlock2(MAout + MMAout)
        
        output = self.FFBlock(MAout)
        output = self.LNBlock3(MAout + output)
        
        return output


class Pooling(nn.Module):
    def __init__(self, dim):
        super(Pooling, self).__init__()
        self.pooling = nn.AdaptiveMaxPool1d(dim)
        
    def forward(self, x: Tensor):
        return self.pooling(x.permute(0, 2, 1)).squeeze(2)


class TextClassification(nn.Module):
    def __init__(self, inputSize: torch.Size, dtype = torch.float32, device = torch.device('cpu')) -> None:
        super(TextClassification, self).__init__()
        
        self.PEBlock = PositionalEncoding(inputSize,
                                          modelDepth = 256,
                                          dtype = dtype,
                                          device = device)

        self.EncodeBlock1 = TransformerEncoder(modelDepth = 256,
                                               numHead = 8,
                                               ffDim = 1024,
                                               dropout = .1,
                                               dtype = dtype,
                                               device = device)
        
        self.DimReduce1 = nn.Linear(256, 128, dtype = dtype, device = device)
        
        self.EncodeBlock2 = TransformerEncoder(modelDepth = 128,
                                               numHead = 8,
                                               ffDim = 512,
                                               dropout = .1,
                                               dtype = dtype,
                                               device = device)
        
        self.DimReduce2 = nn.Linear(128, 64, dtype = dtype, device = device)
        
        self.MABlock3 = MultiheadAttention(modelDepth = 64,
                                           numHeads = 4,
                                           dtype = dtype,
                                           device = device)
        
        self.pooling = Pooling(1)
        
        self.DimReduce3 = nn.Linear(64, 32, dtype = dtype, device = device)
        
        self.DimReduce4 = nn.Linear(32, 14, dtype = dtype, device = device)

    def forward(self, x: Tensor):
        output = self.PEBlock(x)

        output = self.EncodeBlock1(output)
        
        output = self.DimReduce1(output)
        
        output = self.EncodeBlock2(output)
        
        output = self.DimReduce2(output)

        output = self.pooling(output)
        
        output = self.DimReduce3(output)
        
        output = self.DimReduce4(output)

        return output

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

        self.position = PositionalEncoding2D(height = int(numPatches ** .5), width = int(numPatches ** .5), dim = modelDepth)

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