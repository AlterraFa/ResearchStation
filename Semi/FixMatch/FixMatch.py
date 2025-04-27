import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import (DataLoader, TensorDataset, Dataset, ConcatDataset)
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from tqdm.auto import tqdm

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
        transform=weakAugment             # or weakAugment if you want augment on train
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
    unlabeledDS = UnlabeledDataset(
        unlabeled,
        weak_transform = weakAugment,
        strong_transform = strongAugment
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


if __name__ == "__main__":

    device = torch.device('cuda')

    useRatio = .1
    unlabeled = torch.load("./datasets/stl-10/unlabeled.pt"); unlabeled = unlabeled[: int(unlabeled.shape[0] * useRatio)].permute(0, 3, 1, 2)
    train = torch.load("./datasets/stl-10/train.pt"); trainX = train[0].to(torch.float32).permute(0, 3, 1, 2); trainY = train[1].long()
    test = torch.load("./datasets/stl-10/test.pt"); testX = test[0].to(torch.float32).permute(0, 3, 1, 2); testY = test[1].long()
    trainX = trainX / 255
    testX = testX / 255
    unlabeled = unlabeled / 255
    numClasses = len(torch.unique(testY))
    del train, test

    writer = SummaryWriter(log_dir = "FixMatchExperiment")
    depth = 22; width = 2
    model = WRN(depth, width, 10, 0.25)
    model(trainX[:1])
    model.summary()
    model.to(device)
    writer.add_graph(model, trainX[:1].to(device))
    writer.flush()

    
    weakAugment = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

    strongAugment = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops = 3, magnitude = 10),
        T.ToTensor()
    ])


    trainLoader, valLoader, testLoader, unlabeledLoader = allSetAugment(
        [trainX, trainY], 
        [testX, testY], 
        unlabeled, 
        batchSize = 60, 
        muy = 0.7, 
        splitRatio = 0.001
    )
    del trainX, trainY, testX, testY, unlabeled
    
    epochs = 200
    optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, nesterov = True)
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = 20, eta_min = 0)
    supervisedCriterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    unsupervisedCriterion = nn.CrossEntropyLoss()
    earlystop = EarlyStopping(50, 0.00000001, path = f"./Resnet_{depth}_{width}.pt", verbose = True)
    tau = 0.5; l1 = 1e-3; l2 = 1e-3

    trainLosses = []; valLosses = []; 
    pbar = tqdm(range(epochs), desc="Training Epochs")
    for epoch in pbar:
        model.train()
        
        supervisedCost = 0
        consistencyCost = 0
        totalCost = 0
        trainCount = 0
        counter = 0
        for (xBatch, yBatch), (unlabeledWeak, unlabeledStrong) in zip(trainLoader, unlabeledLoader):
            optimizer.zero_grad()

            logits         = model(xBatch.to(device))
            supervisedLoss = supervisedCriterion(logits, yBatch.to(device)).mean()
            distribution   = torch.softmax(logits, dim = 1)
            trainCount     += (torch.argmax(distribution, dim = 1) == yBatch.to(device)).sum().item(); counter += yBatch.shape[0]
            del xBatch, yBatch, logits 

            wLogits            = model(unlabeledWeak.to(device))
            qWeak              = torch.softmax(wLogits, dim = 1)
            confs, pseudoLabel = qWeak.max(dim = 1)
            pseudoLabel        = pseudoLabel.detach()
            mask               = (confs >= tau).float()
            del unlabeledWeak, wLogits, qWeak, confs  

            sLogits          = model(unlabeledStrong.to(device))
            unsupervisedLoss = unsupervisedCriterion(sLogits, pseudoLabel)
            del sLogits, unlabeledStrong
            
            consistencyLoss = (mask * unsupervisedLoss).mean()

            weightParams = [p for n, p in model.named_parameters()
                            if p.requires_grad and "weight" in n]
            l1Norm = sum(p.abs().sum() for p in weightParams)
            l2Norm = sum(p.pow(2.0).sum() for p in weightParams)
            
            loss = supervisedLoss \
                    + consistencyLoss \
                    + l1Norm * l1 \
                    + l2Norm * l2
            loss.backward()
            optimizer.step()


            supervisedCost  += supervisedLoss.item()
            consistencyCost += consistencyLoss.item()
            totalCost       += loss.item()
        
        trainSupLossTotal    = supervisedCost / len(trainLoader)
        consistencyLossTotal = consistencyCost / len(trainLoader)
        totalLoss            = totalCost / len(trainLoader)
        trainAcc             = trainCount / counter

        model.eval()
        runningLoss = 0.0; valCount = 0; counter = 0
        with torch.no_grad():
            for xBatch, yBatch in testLoader:
                xBatch = xBatch.to(device); yBatch = yBatch.to(device)

                outputs      = model(xBatch)
                loss         = supervisedCriterion(outputs, yBatch)
                distribution = torch.softmax(outputs, dim = 1)

                valCount    += (torch.argmax(distribution, dim = 1) == yBatch).sum().item()
                counter     += yBatch.shape[0]
                runningLoss += loss.item()

        valLossTotal = runningLoss / len(testLoader)
        valAcc = valCount / counter

        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']
        
        used     = torch.cuda.memory_allocated()  / 2**20
        reserved = torch.cuda.memory_reserved()   / 2**20

        
        tqdm.write(f"Epoch: {epoch + 1}, Supervised Loss: {trainSupLossTotal:.4f}, Consistency Loss: {consistencyLossTotal:.4f}, Loss: {totalLoss:.4f}, Train Accuracy: {100 * trainAcc:.2f}%, Val loss: {valLossTotal:.4f}, Val Acc: {100 * valAcc:.2f}%")
        pbar.set_postfix({
            "Supervised Loss": f"{trainSupLossTotal:.4f}",
            "Consistency Loss": f"{consistencyLossTotal:.4f}",
            "Loss": f"{totalLoss:.4f}",
            "Val Loss": f"{valLossTotal:.4f}"
        })  
        writer.add_scalar("Loss/Supervised",     trainSupLossTotal,    epoch + 1)
        writer.add_scalar("Loss/Consistency",    consistencyLossTotal, epoch + 1)
        writer.add_scalar("Loss/Total",          totalLoss,            epoch + 1)
        writer.add_scalar("Accuracy/Train",      100 * trainAcc,       epoch + 1)
        writer.add_scalar("Loss/Validation",     valLossTotal,         epoch + 1)
        writer.add_scalar("Accuracy/Validation", 100 * valAcc,         epoch + 1)
        writer.add_scalar("Misc/Lr",             currentLr,            epoch + 1)
        writer.add_scalar("Misc/GPU-used",       used,                 epoch + 1)
        writer.add_scalar("Misc/GPU-reserved",   reserved,             epoch + 1)
        writer.flush()
        writer.close()
        
        earlystop(valLossTotal, model)
        if earlystop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break     