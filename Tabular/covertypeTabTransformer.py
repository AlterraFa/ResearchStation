import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tabTransformer import TabTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.helper import EarlyStopping
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAUROC

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

def split_covtype_data(xCont, xWild, xSoil, y, val_ratio=0.1, test_ratio=0.1, random_state=42):
    x_cat = np.column_stack([xWild, xSoil])

    x_cont_trainval, x_cont_test, \
    x_cat_trainval, x_cat_test, \
    y_trainval, y_test = train_test_split(
        xCont, x_cat, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y
    )

    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    x_cont_train, x_cont_val, \
    x_cat_train, x_cat_val, \
    y_train, y_val = train_test_split(
        x_cont_trainval, x_cat_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_trainval
    )

    return {
        'train': [torch.tensor(x_cont_train, dtype = torch.float32), torch.tensor(x_cat_train, dtype = torch.long), torch.tensor(y_train, dtype = torch.long)],
        'val':   [torch.tensor(x_cont_val, dtype = torch.float32), torch.tensor(x_cat_val, dtype = torch.long), torch.tensor(y_val, dtype = torch.long)],
        'test':  [torch.tensor(x_cont_test, dtype = torch.float32), torch.tensor(x_cat_test, dtype = torch.long), torch.tensor(y_test, dtype = torch.long)],
    }

class CatContDataset(Dataset):
    def __init__(self, data: list[torch.tensor, torch.tensor], label: torch.tensor):
        super().__init__()
        
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.data[0][idx], self.data[1][idx]), self.label[idx]

if __name__ == "__main__":
    
    device = torch.device("cuda")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets/covertype/covtype.data.gz"))
    df = pd.read_csv(root, compression = 'gzip', header = None, nrows = 300000)


    xCont = df.iloc[:, :10].values
    xWild = (df.iloc[:, 10:14].idxmax(axis=1) - 10).values
    xSoil = (df.iloc[:, 14:-1].idxmax(axis=1) - 14).values
    y = (df.iloc[:, -1] - 1).values

    splits = split_covtype_data(xCont, xWild, xSoil, y, val_ratio=0.1, test_ratio=0.1, random_state=42)

    
    trainDs = CatContDataset(splits['train'][:2], splits['train'][2])
    valDs   = CatContDataset(splits['val'][:2], splits['val'][2])
    testDs  = CatContDataset(splits['test'][:2], splits['test'][2])

    trainLoader = DataLoader(trainDs, batch_size = 2048 // 2, shuffle = True, num_workers = 12, persistent_workers = False, pin_memory = True)
    valLoader   = DataLoader(valDs, batch_size = 256, shuffle = False, num_workers = 12, persistent_workers = False, pin_memory = True)
    testLoader  = DataLoader(testDs, batch_size = 256, shuffle = False, num_workers = 12, persistent_workers = False, pin_memory = True)
    
    numClasses = int(splits['train'][2].max() + 1)
    numCont = splits['train'][0].shape[1]
    numCat = [int(splits['train'][1][:, i].max().item()) + 1 for i in range(splits['train'][1].shape[1])]

    model = TabTransformer(numCont    = numCont, 
                           numCat     = numCat, 
                           numClasses = numClasses, 
                           numLayers  = 10,
                           dropout = .12).to(device)
    
    writer = SummaryWriter(log_dir = "runs/covtype/TabTransformer")
    data, _ = next(iter(trainLoader))
    writer.add_graph(model, (data[0].to(device), data[1].to(device)))
    writer.flush()


    epochs = 250; initLR = 5e-3; finalLR = 1e-6; l1 = 1e-3; l2 = 5e-4
    
    classCriterion  = nn.CrossEntropyLoss(label_smoothing = .1, reduction = "mean")
    optimizer       = optim.AdamW(model.parameters(), lr = initLR)
    scheduler       = CosineAnnealingLR(optimizer = optimizer, T_max = epochs, eta_min = finalLR)
    earlyStop       = EarlyStopping(patience = 50, path = f"./runs/covtype/TabTransformer/best.pt", verbose = True)
    auc             = MulticlassAUROC(num_classes = numClasses, average = "macro").to(device)

    pbar = tqdm(range(epochs), desc="Training Epochs", position = 0)
    for epoch in pbar:
        model.train()
        
        trainBar = tqdm(trainLoader, desc = "Train", position = 1, leave = False)
        trainMetrics = {"Loss": 0, "Supervised": 0, "Accuracy": 0, "Correct": 0, "Samples": 0, "AUC": 0}
        for (xCont, xCat), target in trainBar:
            
            optimizer.zero_grad()
            xCont     = xCont.to(device, non_blocking = True)
            xCat      = xCat.to(device, non_blocking = True)
            target    = target.to(device, non_blocking = True)
            batchSize = xCont.shape[0]

            
            logits         = model(xCont, xCat)
            correct        = (logits.argmax(dim=1) == target).sum().item()
            supervisedLoss = classCriterion(logits, target)
            auc.update(torch.softmax(logits, dim=1), target)


            weightParams = [p for n, p in model.named_parameters()
                            if p.requires_grad and "weight" in n]
            l1Norm = sum(p.abs().sum() for p in weightParams)
            l2Norm = sum(p.pow(2.0).sum() for p in weightParams)
            

            loss = supervisedLoss + l1 * l1Norm + l2 * l2Norm


            trainMetrics["Samples"]    += batchSize
            trainMetrics["Loss"]       += loss.item()
            trainMetrics["Supervised"] += supervisedLoss.item()
            trainMetrics["Correct"]    += correct

            trainBar.set_postfix({
                "Loss": f"{trainMetrics['Loss']/(trainBar.n+1):.3f}",
                "Supervised": f"{trainMetrics['Supervised']/(trainBar.n+1):.3f}",
                "Correct": f"{trainMetrics['Correct']}/{trainMetrics['Samples']}",
                "Accuracy": f"{100 * trainMetrics['Correct'] / trainMetrics['Samples']:.2f}%",
            })
            
            loss.backward()
            optimizer.step()
            
         
        trainMetrics["Supervised"] /= len(trainLoader)
        trainMetrics["Loss"]       /= len(trainLoader)
        trainMetrics["Accuracy"]   = 100 * trainMetrics["Correct"] / trainMetrics["Samples"]
        trainMetrics["AUC"]        = auc.compute().item()
        auc.reset()
        

        with torch.no_grad():
            valBar = tqdm(valLoader, desc = "Val", position = 2, leave = False)
            valMetrics = {"Supervised": 0, "Accuracy": 0, "Samples": 0, "Correct": 0, "AUC": 0}
            for (xCont, xCat), target in valBar:
                xCont     = xCont.to(device, non_blocking = True)
                xCat      = xCat.to(device, non_blocking = True)
                target    = target.to(device, non_blocking = True)
                batchSize = xCont.shape[0]


                logits         = model(xCont, xCat)
                correct        = (logits.argmax(dim=1) == target).sum().item()
                supervisedLoss = classCriterion(logits, target)
                auc.update(torch.softmax(logits, dim=1), target)


                valMetrics["Samples"]    += batchSize
                valMetrics["Supervised"] += supervisedLoss.item()
                valMetrics["Correct"]    += correct

                valBar.set_postfix({
                    "Supervised": f"{valMetrics['Supervised'] / (valBar.n+1):.3f}",
                    "Accuracy": f"{100 * valMetrics['Correct'] / valMetrics['Samples']:.2f}%",
                    "Correct": f"{valMetrics['Correct']}/{valMetrics['Samples']}",
                })

            valMetrics['Supervised'] /= len(valLoader)
            valMetrics['Accuracy']   = 100 * valMetrics['Correct'] / valMetrics['Samples']
            valMetrics['AUC']        = auc.compute().item()
            auc.reset()

        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']
        
        used     = torch.cuda.memory_allocated()  / 2**20
        reserved = torch.cuda.memory_reserved()   / 2**20

        tqdm.write(
            f"Epoch {epoch+1}/{epochs} — "
            f"Sup Train: {trainMetrics['Supervised']:.4f}, "
            f"Acc Train: {trainMetrics['Accuracy']:.2f}%, "
            f"AUC Train: {trainMetrics['AUC']:.4f}, "
            f"Sup Val: {valMetrics['Supervised']:.4f}, "
            f"Acc Val: {valMetrics['Accuracy']:.2f}%, "
            f"AUC Val: {valMetrics['AUC']:.4f}, "
            f"No update: {earlyStop.counter}/{earlyStop.patience}"
        )

        writer.add_scalar("Loss/Supervised Train",  trainMetrics["Supervised"], epoch+1)
        writer.add_scalar("Loss/Supervised Val",    valMetrics["Supervised"],   epoch+1)
        writer.add_scalar("Metrics/Train/Accuracy", trainMetrics["Accuracy"],   epoch+1)
        writer.add_scalar("Metrics/Train/AUC",      trainMetrics["AUC"],        epoch+1)
        writer.add_scalar("Metrics/Val/Accuracy",   valMetrics["Accuracy"],     epoch+1)
        writer.add_scalar("Metrics/Val/AUC",        valMetrics["AUC"],          epoch+1)
        writer.add_scalar("Misc/LearningRate",      currentLr,                  epoch+1)
        writer.add_scalar("Misc/Memory/Allocated",  used,                       epoch+1)
        writer.add_scalar("Misc/Memory/Reserved",   reserved,                   epoch+1)
        writer.flush()

        earlyStop(valMetrics['Supervised'], model)
        if earlyStop.early_stop:
            tqdm.write("Early stopping triggered.")