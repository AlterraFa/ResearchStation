import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR

from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
from tqdm.auto import tqdm

from models import *



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

    


    trainLoader, valLoader, testLoader, unlabeledLoader = allSetAugment(
        [trainX, trainY], 
        [testX, testY], 
        unlabeled, 
        batchSize = 40, 
        muy = 0.7, 
        splitRatio = 0.1
    )
    del trainX, testX, testY, unlabeled

    initLR = 1e-3; targetLR = 1e-10
    epochs = 200; tau = 0.5; l1 = 1e-3; l2 = 1e-3; reAugmentApply = 3; 
        
    optimizer             = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, nesterov = True)
    cosine                = CosineAnnealingLR(optimizer = optimizer, T_max = epochs // 2, eta_min = targetLR)
    constant              = ConstantLR(optimizer = optimizer, factor = targetLR / initLR, total_iters = 1)
    scheduler             = SequentialLR(
        optimizer = optimizer,
        schedulers = [cosine, constant],
        milestones = [epochs // 2]
    )
    supervisedCriterion   = nn.CrossEntropyLoss(label_smoothing = 0.1)
    unsupervisedCriterion = nn.CrossEntropyLoss(reduction = 'none')
    alignment             = DistributionAlignment(trainY, numClasses = numClasses, momentum = 0.999).to(device)
    earlystop             = EarlyStopping(50, 0.00000001, path = f"./Resnet_{depth}_{width}.pt", verbose = True)


    trainLosses = []; valLosses = []; 
    pbar = tqdm(range(epochs), desc="Training Epochs")
    for epoch in pbar:
        model.train()
        
        supervisedCost = 0
        consistencyCost = 0
        totalCost = 0
        trainCount = 0
        counter = 0
        for (xBatch, yBatch), unlabeled in zip(trainLoader, cycle(unlabeledLoader)):
            unlabeled = unlabeled[0]
            optimizer.zero_grad()

            logits         = model(xBatch.to(device))
            supervisedLoss = supervisedCriterion(logits, yBatch.to(device)).mean()
            distribution   = torch.softmax(logits, dim = 1)
            trainCount     += (torch.argmax(distribution, dim = 1) == yBatch.to(device)).sum().item(); 
            counter        += yBatch.shape[0]
            del xBatch, yBatch, logits 

            with torch.no_grad():
                unlabeledWeak = torch.stack([
                    weakAugment(img) for img in unlabeled
                ])
                wLogits            = model(unlabeledWeak.to(device))
                qWeak              = torch.softmax(wLogits, dim = 1)
                confs, _           = qWeak.max(dim = 1)
                pseudoLabel        = alignment(qWeak)
                
                mask               = (confs >= tau).float()
                del unlabeledWeak, wLogits, qWeak, confs  

            unsupervisedLosses = 0.0
            for _ in range(reAugmentApply):
                unlabeledStrong    = torch.stack([strongAugment(img) for img in unlabeled])
                unlabeledStrong    = unlabeledStrong.to(device)
                sLogits            = model(unlabeledStrong)
                scalarLoss         = (mask * unsupervisedCriterion(sLogits, pseudoLabel)).mean()
                unsupervisedLosses += scalarLoss
                del sLogits, unlabeledStrong
            
            consistencyLoss = unsupervisedLosses / reAugmentApply

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
            for xBatch, yBatch in valLoader:
                xBatch = xBatch.to(device); yBatch = yBatch.to(device)

                outputs      = model(xBatch)
                loss         = supervisedCriterion(outputs, yBatch)
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
        
        earlystop(valLossTotal, model)
        if earlystop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break     