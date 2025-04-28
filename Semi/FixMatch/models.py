import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import (DataLoader, TensorDataset, Dataset, ConcatDataset)

from typing import Tuple

class EarlyStopping:
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0, 
                 path: str = "checkpoint.pt",
                 verbose: bool = False):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = torch.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module):
        # check if loss improved by at least min_delta
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}. Saved model to {self.path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in val loss for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float = 0.0):
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

class BlockStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float, numBlock: int):
        super(BlockStack, self).__init__() 
        self.group = self.make_group(in_channels, out_channels, stride, dropout, numBlock)
        
    def make_group(self, in_channels: int, out_channels: int, stride: int, dropout: float, numBlock: int):
        layers = []
        for idx in range(numBlock):
            if idx == 0:
                layers.append(ResnetBlock(in_channels, out_channels, stride, dropout))
            else:
                layers.append(ResnetBlock(out_channels, out_channels, 1, dropout))
        
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.group(x)

        

class WRN(nn.Module):
    def __init__(self, depth: int, widenFact: int, numClasses: int, dropout: float = 0.0):
        super(WRN, self).__init__()
        assert (depth - 4) % 6 == 0
        numBlock = (depth - 4) // 6

        channelDepth = [16, 16 * widenFact, 32 * widenFact, 64 * widenFact]
        strides = [1, 2, 2]

        self.stem = nn.Conv2d(3, channelDepth[0], kernel_size = 3, stride = 1)
        
        self.largeGroup = nn.ModuleList(
            [BlockStack(channelDepth[i], channelDepth[i + 1], strides[i], dropout, numBlock) for i in range(3)]
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


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor = None, transform=None):
        if labels is not None:
            assert images.shape[0] == labels.shape[0] 
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]        # shape [C,H,W]
        if self.labels is not None:
            y = self.labels[idx]        # one‑hot float
        if self.transform:
            x = self.transform(x)
        return x, y if self.labels is not None else x
    
class UnlabeledDataset(Dataset):
    def __init__(self, images: torch.Tensor, weak_transform=None, strong_transform=None):
        self.images = images
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]  # shape [C, H, W]
        x_weak = self.weak_transform(x) if self.weak_transform else x
        x_strong = self.strong_transform(x) if self.strong_transform else x
        return x_weak, x_strong

weakAugment = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=(96, 96), padding=8, pad_if_needed=True),
    T.ToTensor()
])

strongAugment = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops = 3, magnitude = 10),
    T.ToTensor()
])

def allSetAugment(
    train: Tuple[torch.Tensor, torch.Tensor],
    test: Tuple[torch.Tensor, torch.Tensor],
    unlabeled: torch.Tensor,
    batchSize: int,
    muy: float,
    splitRatio: float = 0.1
):
    train_images, train_labels = train
    num_train = train_images.shape[0]
    split = int(num_train * splitRatio)

    # split off a validation slice
    trainDS = TensorImageDataset(
        train_images[split:], 
        train_labels[split:].long(),
        transform=weakAugment
    )
    valDS   = TensorImageDataset(
        train_images[:split],
        train_labels[:split].long(),
        transform=None
    )
    testDS  = TensorImageDataset(
        test[0],
        test[1].long(),
        transform=None
    )
    unlabeledDS = TensorImageDataset(
        unlabeled
    )

    trainLoader     = DataLoader(trainDS, batch_size=batchSize, shuffle=True)
    valLoader       = DataLoader(valDS,   batch_size=batchSize, shuffle=True)
    testLoader      = DataLoader(testDS,  batch_size=batchSize, shuffle=False)
    unlabeledLoader = DataLoader(
        unlabeledDS, 
        batch_size=int(muy * batchSize), 
        shuffle=True
    )

    return trainLoader, valLoader, testLoader, unlabeledLoader

class DistributionAlignment(nn.Module):
    def __init__(self, labels: torch.tensor, numClasses: int, momentum: float):
        super(DistributionAlignment, self).__init__()
        
        counts = torch.bincount(labels)
        pEmperical = (counts.float() / labels.numel())
        
        self.register_buffer("pEmperical", pEmperical)
        self.register_buffer("pRunning", torch.zeros(numClasses))
        self.momentum = momentum
        
    def forward(self, q: torch.Tensor):
        pBatch = q.mean(dim = 0)
        self.pRunning = (
            self.momentum * self.pRunning 
            + (1 - self.momentum) * pBatch
        )
        
        labelTilde = q * (self.pEmperical / (self.pRunning + 1e-6)).unsqueeze(0)
        
        return labelTilde / labelTilde.sum(dim = 1, keepdim = True)