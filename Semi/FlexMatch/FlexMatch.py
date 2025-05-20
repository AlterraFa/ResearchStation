import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from utils.WRN import *
from utils.helper import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR

from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
from tqdm.auto import tqdm

if __name__ == "__main__":

    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    torch.manual_seed(45)

    useRatio = .01
    unlabeled = torch.load("./datasets/stl-10/unlabeled.pt"); unlabeled = unlabeled[: int(unlabeled.shape[0] * useRatio)].permute(0, 3, 1, 2)
    train = torch.load("./datasets/stl-10/train.pt"); trainX = train[0].to(torch.float32).permute(0, 3, 1, 2); trainY = train[1].long()
    trainX = trainX.float().div_(255.0)
    trainX.sub_(0.5).div_(0.5)
    unlabeled = unlabeled.float().div_(255.0)
    unlabeled.sub_(0.5).div_(0.5)
    classArray = torch.unique(trainY).to(gpu).unsqueeze(0)
    numClasses = classArray.shape[1]
    
    del train
    (trainX, trainY), (valX, valY) = valSplit((trainX, trainY), 0.1)

    writer = SummaryWriter(log_dir = "FixMatchExperiment")
    depth = 22; width = 2
    model = WRN(ResnetBlock, depth, width, 10, 2, 0.25)
    model(trainX[:1])
    model.summary()
    model.to(gpu)
    writer.add_graph(model, trainX[:1].to(gpu))
    writer.flush()

    
    reaugmentApply = 2
    trainDS     = VariableTensorDataset(trainX, trainY, augments = [weakAugment])
    valDS       = VariableTensorDataset(valX,   valY,   augments = None)
    unlabeledDS = VariableTensorDataset(unlabeled, augments = [noAugment, weakAugment] + [strongAugment] * reaugmentApply, release = (1,) + tuple(range(2, 2 + reaugmentApply)))

    trainSampleSz = len(trainDS); valSampleSz = len(valDS)

    batchSize = 64; muy = 1
    trainLoader     = DataLoader(trainDS, batch_size = batchSize, shuffle = True, num_workers = 4, pin_memory = True, persistent_workers = True)
    unlabeledLoader = DataLoader(unlabeledDS, batch_size = int(muy * batchSize), shuffle = True, num_workers = 4, pin_memory = True, persistent_workers = True)
    valLoader       = DataLoader(valDS, batch_size = 32, shuffle = True, num_workers = 4, pin_memory = True, persistent_workers = True)

    

    initLR = 1e-3; targetLR = 1e-10
    epochs = 500; tau = 0.5; l1 = 1e-3; l2 = 1e-4; 

    optimizer             = optim.AdamW(model.parameters(), lr = initLR, betas = (0.95, 0.999))
    scheduler             = CosineAnnealingLR(optimizer = optimizer, T_max = 25, eta_min = targetLR)
    supervisedCriterion   = nn.CrossEntropyLoss(label_smoothing = 0.1)
    unsupervisedCriterion = nn.CrossEntropyLoss()
    earlystop             = EarlyStopping(50, 0.00000001, path = f"./Resnet_{depth}_{width}.pt", verbose = True)
    alignment             = DistributionAlignment(trainY, numClasses, momentum = 0.995).to(gpu)
    thresh                = VariableThresh(len(unlabeledDS), numClasses).to(gpu)

    pbar = tqdm(range(epochs), desc="Training Epochs", position = 0)
    for epoch in pbar:
        model.train()
        
        trainBar = tqdm(trainLoader, desc = "Train", position = 1, leave = False); trainCnt = 0
        trainMetrics = {"Total": 0, "Supervised": 0, "Consistency": 0, "Accuracy": 0}
        for (xBatch, yBatch), ((unlabeledWeak, *unlabeledStrongList), indices) in zip(trainBar, cycle(unlabeledLoader)):
            optimizer.zero_grad()
            xBatch = xBatch.to(gpu)
            yBatch = yBatch.to(gpu)

            logits         = model(xBatch)
            supervisedLoss = supervisedCriterion(logits, yBatch).mean()
            distribution   = torch.softmax(logits, dim = 1)
            correct        = (torch.argmax(distribution, dim = 1) == yBatch).sum(); trainCnt += yBatch.shape[0]

            with torch.no_grad():
                unlabeledWeak     = unlabeledWeak.to(gpu)
                wLogits           = model(unlabeledWeak.to(gpu))
                qWeak             = torch.softmax(wLogits, dim = 1)
                mask, pseudoLabel = thresh(qWeak, indices)

            
            consecLoss = 0 # Augmentation anchoring
            for unlabeledStrong in unlabeledStrongList:
                unlabeledStrong = unlabeledStrong.to(gpu, non_blocking = True)

                sLogits      = model(unlabeledStrong)
                scalarLoss   = (mask * unsupervisedCriterion(sLogits, pseudoLabel)).mean()
                consecLoss  += scalarLoss

                
            consistencyLoss = consecLoss / reaugmentApply

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


            trainMetrics["Consistency"] += consistencyLoss.item()
            trainMetrics["Total"]       += loss.item()
            trainMetrics["Supervised"]  += supervisedLoss.item()
            trainMetrics["Accuracy"]    += correct.item()
            
            trainBar.set_postfix({
                "T": f"{trainMetrics['Total']/ (trainBar.n+1):.3f}",
                "S": f"{trainMetrics['Supervised']/(trainBar.n+1):.3f}",
                "C": f"{trainMetrics['Consistency']/(trainBar.n+1):.3f}",
                "Acc": f"{trainMetrics['Accuracy']/(trainCnt) * 100:.2f}%",
            })
        
        trainMetrics["Consistency"] /= len(trainLoader)
        trainMetrics["Total"]       /= len(trainLoader)
        trainMetrics["Supervised"]  /= len(trainLoader)
        trainMetrics["Accuracy"]    /= trainSampleSz

        with torch.no_grad():
            valBar = tqdm(valLoader, desc = "Val", position = 2, leave = False)
            valMetrics = {"Accuracy": 0, "Cost": 0}; valCnt = 0
            for (xBatch, yBatch) in valBar:
                xBatch = xBatch.to(gpu)
                yBatch = yBatch.to(gpu)

                supervisedLogits = model(xBatch)
                supervisedLoss   = supervisedCriterion(supervisedLogits, yBatch)
                supervisedDist   = torch.softmax(supervisedLogits, dim = 1) 

                valMetrics['Accuracy'] += (torch.argmax(supervisedDist, dim = 1) == yBatch).sum().item()
                valMetrics["Cost"]     += supervisedLoss.item()
                
                valCnt += yBatch.shape[0]
                
                valBar.set_postfix({
                    "Acc": f"{valMetrics['Accuracy'] / (valCnt) * 100:.3f}%",
                    "Cost": f"{valMetrics['Cost'] / (valBar.n+1):.3f}"
                })

            valMetrics['Cost']     /= len(valLoader)
            valMetrics['Accuracy'] /= valSampleSz


        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']
        
        used     = torch.cuda.memory_allocated()  / 2**20
        reserved = torch.cuda.memory_reserved()   / 2**20

        
        tqdm.write(
            f"Epoch {epoch+1}/{epochs} — "
            f"Sup: {trainMetrics['Supervised']:.4f}, "
            f"Cons: {trainMetrics['Consistency']:.4f}, "
            f"Total: {trainMetrics['Total']:.4f}, "
            f"Train Acc: {100*trainMetrics['Accuracy']:.2f}%, "
            f"Val Loss: {valMetrics['Cost']:.4f}, "
            f"Val Acc: {100*valMetrics['Accuracy']:.2f}%, "
            f"LR: {currentLr:.1e}, "
            f"No update: {earlystop.counter}/{earlystop.patience}"
        )
        writer.add_scalar("Loss/Supervised",     trainMetrics["Supervised"], epoch+1)
        writer.add_scalar("Loss/Consistency",    trainMetrics["Consistency"], epoch+1)
        writer.add_scalar("Loss/Total",          trainMetrics["Total"],       epoch+1)
        writer.add_scalar("Accuracy/Train",      100*trainMetrics["Accuracy"],epoch+1)
        writer.add_scalar("Loss/Validation",     valMetrics["Cost"],          epoch+1)
        writer.add_scalar("Accuracy/Validation", 100*valMetrics["Accuracy"],  epoch+1)
        writer.add_scalar("Misc/LearningRate",   currentLr,                    epoch+1)
        writer.flush()
        
        earlystop(valMetrics['Cost'], model)
        if earlystop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break