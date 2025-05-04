import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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