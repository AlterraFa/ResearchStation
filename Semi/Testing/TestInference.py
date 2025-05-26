import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

import torch
from torchvision import datasets, transforms, models

from utils.transformer import Transformer, SpatialEncoding
from utils.imageTransformer import DeTr

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    
    data = torch.zeros((2, 3, 640, 640))
    
    testing = DeTr(numClasses = 10, nHeads = 8, nEncoders = 5, nDecoders = 5, hiddenDim = 512).to(gpu)
    testing.summary()
    testing(data.to(gpu), True)
    
    writer = SummaryWriter()
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    writer.add_graph(testing, data.to(gpu))
    writer.flush()
    writer.close()