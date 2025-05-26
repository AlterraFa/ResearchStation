import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR, CosineAnnealingWarmRestarts
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, TensorDataset

from utils.WRN import WRN, ConvNeXt, ResnetBlock
from utils.transformer import ViT
from utils.helper import EarlyStopping

from tqdm.auto import tqdm

normTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == "__main__":
    device = torch.device('cuda')
    
    # train = torch.load("./datasets/stl-10/train.pt"); trainX = train[0].to(torch.float32).permute(0, 3, 1, 2); trainY = train[1].long()
    # test = torch.load("./datasets/stl-10/test.pt"); testX = test[0].to(torch.float32).permute(0, 3, 1, 2); testY = test[1].long()
    # trainX = torch.cat([normTransform(img) for img in trainX])
    # testX = torch.cat([normTransform(img) for img in testX])

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train = datasets.CIFAR10(
        root = "datasets",
        train = True,
        download = True,
        transform = transform
    )
    
    test = datasets.CIFAR10(
        root = "datasets",
        train = False,
        download = True,
        transform = transform
    )
    
    valLength   = int(0.1 * len(train))
    trainLength = len(train) - valLength
    train, val = random_split(
        dataset = train, 
        lengths = [trainLength, valLength],
        generator = torch.Generator().manual_seed(42)
    )
    
    
    
    trainLoader = DataLoader(train, batch_size = 64, shuffle = True, num_workers = 12, pin_memory = True)
    valLoader   = DataLoader(val, batch_size = 1000, shuffle = True, num_workers = 12, pin_memory = True)

    widenFact = 4; depth = 28
    model = WRN(blockType = ConvNeXt, depth = depth, widenFact = widenFact, numClasses = 10, dropout = .05, patchSz = 2).to(device)
    
    
    epochs = 400; switchEpoch = 60; initLR = 7.5e-4; finalLR = 1e-10; l1 = 1e-5; l2 = 1e-5
    criterion = nn.CrossEntropyLoss(label_smoothing = .1)
    optimizer = optim.AdamW(model.parameters(), lr = initLR, betas = (0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = epochs, eta_min = finalLR)
    # constant  = ConstantLR(optimizer = optimizer, factor = finalLR / initLR, last_epoch = -1, total_iters = epochs - switchEpoch)
    # scheduler = SequentialLR(
    #     optimizer = optimizer,
    #     schedulers = [cosine, constant],
    #     milestones = [switchEpoch]
    # )
    earlyStop = EarlyStopping(patience = 50, path = f"./WRN_{depth}_{widenFact}.pt", verbose = True)
    

    trainLosses = []; valLosses = []; 
    pbar = tqdm(range(epochs), desc="Training Epochs")
    for epoch in pbar:
        optimizer.zero_grad()
        model.train()
        
        supervisedCost = 0
        totalCost = 0
        trainCount = 0
        counter = 0
        for (xBatch, yBatch) in trainLoader:
            xBatch = xBatch.to(device, non_blocking=True)
            yBatch = yBatch.to(device, non_blocking=True)
            

            logits         = model(xBatch)
            supervisedLoss = criterion(logits, yBatch).mean()
            distribution   = torch.softmax(logits, dim = 1)
            trainCount     += (torch.argmax(distribution, dim = 1) == yBatch).sum().item(); 
            counter        += yBatch.shape[0]
            # del xBatch, yBatch, logits 


            weightParams = [p for n, p in model.named_parameters()
                            if p.requires_grad and "weight" in n]
            l1Norm = sum(p.abs().sum() for p in weightParams)
            l2Norm = sum(p.pow(2.0).sum() for p in weightParams)
            
            loss = supervisedLoss \
                    + l1Norm * l1 \
                    + l2Norm * l2
            loss.backward()
            optimizer.step()


            supervisedCost  += supervisedLoss.item()
            totalCost       += loss.item()
        
        trainSupLossTotal    = supervisedCost / len(trainLoader)
        totalLoss            = totalCost / len(trainLoader)
        trainAcc             = trainCount / counter

        model.eval()
        runningLoss = 0.0; valCount = 0; counter = 0
        with torch.no_grad():
            for xBatch, yBatch in valLoader:
                xBatch = xBatch.to(device, non_blocking=True)
                yBatch = yBatch.to(device, non_blocking=True)

                outputs      = model(xBatch)
                loss         = criterion(outputs, yBatch)
                distribution = torch.softmax(outputs, dim = 1)

                valCount    += (torch.argmax(distribution, dim = 1) == yBatch).sum().item()
                counter     += yBatch.shape[0]
                runningLoss += loss.item()

        valLossTotal = runningLoss / len(valLoader)
        valAcc = valCount / counter

        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']
        
        used     = torch.cuda.memory_allocated()  / 2**20
        reserved = torch.cuda.memory_reserved()   / 2**20

        
        tqdm.write(f"Epoch: {epoch + 1}, Supervised Loss: {trainSupLossTotal:.4f}, Loss: {totalLoss:.4f}, Train Accuracy: {100 * trainAcc:.2f}%, Val loss: {valLossTotal:.4f}, Val Acc: {100 * valAcc:.2f}%, currentLR: {currentLr}")
        pbar.set_postfix({
            "Supervised Loss": f"{trainSupLossTotal:.4f}",
            "Loss": f"{totalLoss:.4f}",
            "Val Loss": f"{valLossTotal:.4f}"
        })  
        
        earlyStop(valLossTotal, model)
        if earlyStop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break     

