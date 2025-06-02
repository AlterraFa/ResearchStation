import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAUROC

from utils.SAINT import SAINT
from utils.helper import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm
from joblib import dump


class CutMixTabular(Dataset):
    def __init__(self, contValues, catValues, labels, mixRatio = .5):
        super(CutMixTabular, self).__init__()
        
        self.contValues = torch.Tensor(contValues).to(torch.float)
        self.catValues = torch.Tensor(catValues).to(torch.long)
        self.labels = torch.Tensor(labels).to(torch.long)
        
        self.contFeatSize = contValues.shape[1]
        self.catFeatSize = catValues.shape[1]
        
        self.mixRatio = mixRatio
        self.pretrainMode = False

    def __len__(self):
        return self.labels.shape[0]
        
    def __getitem__(self, index):

        if self.pretrainMode == False:
            return self.contValues[index], self.catValues[index], self.labels[index]
        
        origSample   = torch.cat([self.contValues[index], self.catValues[index]])
        randIdx      = torch.randint(0, self.__len__(), (1, )).item()
        randomSample = torch.cat([self.contValues[randIdx], self.catValues[randIdx]])
        
        sampleMask = torch.bernoulli(torch.full((self.contFeatSize + self.catFeatSize, ), self.mixRatio))
        
        augmentedSample = origSample * (1 - sampleMask) + randomSample * sampleMask
        
        return self.contValues[index], self.catValues[index], \
               augmentedSample[: self.contFeatSize].to(torch.float), augmentedSample[self.contFeatSize: ].to(torch.long)

def readKDD(root: str, nrows = None):
    df = pd.read_csv(root + 'kddcup.data_10_percent_corrected', header=None, nrows = nrows)

    with open(root + 'kddcup.names', 'r') as file:
        lines = file.readlines()
        columnNames = []; valueType = []
        for line in lines:
            columnNames += [line.split(": ")[0]]
            valueType   += [line.split(": ")[1][:-2]]
        valueType.pop(-1)
        
    df.columns = columnNames
    valueType  = np.array(valueType)
    typeMask   = valueType == 'continuous'

    gtValues   = df.iloc[:, -1].values; df = df.drop('labels', axis = 1)
    contValues = df.iloc[:, typeMask].values
    catValues  = df.iloc[:, ~typeMask].values
    
    le       = LabelEncoder()
    gtValues = le.fit_transform(gtValues)
    catCount = []; validCol = []
    for idx in range(catValues.shape[1]):
        le = LabelEncoder()
        encoded = le.fit_transform(catValues[:, idx])
        uniqueCount = len(np.unique(catValues[:, idx]))

        if uniqueCount >= 2:
            catValues[:, idx] = encoded
            catCount += [len(np.unique(catValues[:, idx]))]
            validCol += [idx]

    catValues  = catValues[:, validCol].astype(int)
    numCont    = contValues.shape[1]
    numClasses = len(np.unique(gtValues))

    return (catValues, catCount), (contValues, numCont), (gtValues, numClasses)

def mixup(x1, x2, alpha=1.0):
    ''' Applies mixup between two embeddings (x1, x2) '''
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    return lam * x1 + (1 - lam) * x2, lam

def infoNCE(view1: torch.Tensor, view2: torch.Tensor, temperature = 0.1):
    B, N, D = view1.shape

    view1 = F.normalize(view1.mean(dim = 1), dim = 1)
    view2 = F.normalize(view2.mean(dim = 1), dim = 1)

    logits  = torch.matmul(view1, view2.T) / temperature
    targets = torch.arange(0, B, 1, device = view1.device)
    loss    = F.cross_entropy(logits, targets)

    return loss
    

root = "./datasets/KDD/" 
if __name__ == "__main__":    
    device = torch.device('cuda')

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/covertype/covtype.data.gz"))
    df = pd.read_csv(root, compression = 'gzip', header = None, nrows = 300000)


    xCont = df.iloc[:, :10].values
    xWild = (df.iloc[:, 10:14].idxmax(axis=1) - 10).values
    xSoil = (df.iloc[:, 14:-1].idxmax(axis=1) - 14).values
    y = (df.iloc[:, -1] - 1).values

    contValues = xCont
    catValues  = np.concatenate([xWild[..., None], xSoil[..., None]], axis = 1)
    gtValues   = y

    numClasses = len(np.unique(gtValues))
    numCont    = contValues.shape[1]
    catCount   = [len(np.unique(catValues[:, idx])) for idx in range(catValues.shape[1])]
    
    
    contTrain, contTest, \
    catTrain, catTest, \
    gtTrain, gtTest = train_test_split(contValues, catValues, gtValues, random_state = 42, stratify = gtValues, test_size = .1)
    
    contTrain, contVal, \
    catTrain, catVal, \
    gtTrain, gtVal = train_test_split(contTrain, catTrain, gtTrain, random_state = 42, stratify = gtTrain, test_size = 0.15)
    
    scaler = StandardScaler()
    contTrain = scaler.fit_transform(contTrain)
    contVal = scaler.transform(contVal)
    contTest = scaler.transform(contTest)

    
    trainDs = CutMixTabular(contValues = contTrain, catValues = catTrain, labels = gtTrain, mixRatio = 0.3)
    valDs   = CutMixTabular(contValues = contVal, catValues = catVal, labels = gtVal, mixRatio = .3)
    trainLoader = DataLoader(trainDs, batch_size = 350, shuffle = True, num_workers = 12, persistent_workers = True, pin_memory = True); trainLoader.dataset.pretrainMode = True
    valLoader   = DataLoader(valDs, batch_size = 200, shuffle = True, num_workers = 12, persistent_workers = True, pin_memory = True); valLoader.dataset.pretrainMode = True
    data = next(iter(trainLoader))
    
    model = SAINT(numCont = numCont, catCount = catCount, numClasses = numClasses, numStages = 6, dropout = 0.0).to(device)
    embedding = model.get_embedding(data[0].to(device), data[1].to(device), pretrainMode = True)
    model(embedding = embedding, pretrainMode = True)

    writeRoot = f"./runs/covtype/{model.__class__.__name__}/pretrained"
    writer = SummaryWriter(writeRoot)
    writer.add_graph(model, (data[0].to(device), data[1].to(device)))
    writer.flush()
    dump(scaler, f"{writeRoot}/scaler.pkl")
    
    
    epochs = 50; initLR = 2e-4; finalLR = 1e-6; l1 = 1e-4; l2 = 1e-4
    lambdaCont = 5.0; lambdaCat = 1.0
    
    optimizer       = optim.AdamW(model.parameters(), lr = initLR)
    scheduler       = CosineAnnealingLR(optimizer = optimizer, T_max = epochs, eta_min = finalLR)
    earlyStop       = EarlyStopping(patience = 20, path = f"{writeRoot}/pretrainedBest.pt", verbose = True)
    auc             = MulticlassAUROC(num_classes = numClasses, average = "macro").to(device)
    
    pbar = tqdm(range(epochs), desc="Training Epochs", position = 0)
    for epoch in pbar:
        model.train()
        
        trainBar = tqdm(trainLoader, desc = "Train", position = 1, leave = False)
        trainMetrics = {"Loss": 0, "Contrastive": 0, "DenoisingCont": 0, "DenoisingCat": 0}
        for view1Cont, view1Cat, view2Cont, view2Cat in trainBar:
            optimizer.zero_grad()
            
            view1Cont = view1Cont.to(device, non_blocking = True); view1Cat = view1Cat.to(device, non_blocking = True)
            view2Cont = view2Cont.to(device, non_blocking = True); view2Cat = view2Cat.to(device, non_blocking = True)
            
            view1Embedding = model.get_embedding(view1Cont, view1Cat, pretrainMode = True)
            view2Embedding = model.get_embedding(view2Cont, view2Cat, pretrainMode = True)
            mixedEmbedding, _ = mixup(view1Embedding, view2Embedding, 0.75)
            
            contrast1, _, *_ = model(embedding = view1Embedding, pretrainMode = True)
            contrast2, cont2, cat2 = model(embedding = mixedEmbedding, pretrainMode = True)
           
            contrastiveLoss = infoNCE(contrast1, contrast2) 

            denoisingCont = 0.0
            for idx, pred in enumerate(cont2):
                target = view1Cont[:, idx].unsqueeze(1)
                denoisingCont += F.smooth_l1_loss(pred, target)
            denoisingCont /= idx + 1

            denoisingCat = 0.0
            for idx, pred in enumerate(cat2):
                target = view1Cat[:, idx]
                denoisingCat += F.cross_entropy(pred, target) 
            denoisingCat /= idx + 1

            weightParams = [p for n, p in model.named_parameters()
                            if p.requires_grad and "weight" in n]
            l1Norm = sum(p.abs().sum() for p in weightParams)
            l2Norm = sum(p.pow(2.0).sum() for p in weightParams)
            
            loss = contrastiveLoss \
                + denoisingCat * lambdaCat \
                + denoisingCont * lambdaCont \
                + l1Norm * l1 \
                + l2Norm * l2 

            trainMetrics["Loss"]          += loss
            trainMetrics['Contrastive']   += contrastiveLoss
            trainMetrics['DenoisingCont'] += denoisingCont
            trainMetrics['DenoisingCat']  += denoisingCat
            
            trainBar.set_postfix({
                "Loss": f"{trainMetrics['Loss']/(trainBar.n+1):.3f}",
                "Contrastive": f"{trainMetrics['Contrastive']/(trainBar.n + 1):.3f}",
                "Cont Denoise": f"{trainMetrics['DenoisingCont']/(trainBar.n + 1):.3f}",
                "Cat Denoise": f"{trainMetrics['DenoisingCat']/(trainBar.n + 1):.3f}"
            })
            
            loss.backward()
            optimizer.step()
        trainMetrics['Loss']          /= (trainBar.n + 1)
        trainMetrics['Contrastive']   /= (trainBar.n + 1)
        trainMetrics['DenoisingCont'] /= (trainBar.n + 1)
        trainMetrics['DenoisingCat']  /= (trainBar.n + 1)
        
        
        with torch.no_grad():
            valBar = tqdm(valLoader, desc = "Val", position = 2, leave = False)
            valMetrics = {"Loss": 0, "Contrastive": 0, "DenoisingCont": 0, "DenoisingCat": 0}
            for view1Cont, view1Cat, view2Cont, view2Cat in valBar:

                view1Cont = view1Cont.to(device, non_blocking = True); view1Cat = view1Cat.to(device, non_blocking = True)
                view2Cont = view2Cont.to(device, non_blocking = True); view2Cat = view2Cat.to(device, non_blocking = True)
                
                view1Embedding = model.get_embedding(view1Cont, view1Cat, pretrainMode = True)
                view2Embedding = model.get_embedding(view2Cont, view2Cat, pretrainMode = True)
                mixedEmbedding, _ = mixup(view1Embedding, view2Embedding, 0.75)
                
                contrast1, _, *_ = model(embedding = view1Embedding, pretrainMode = True)
                contrast2, cont2, cat2 = model(embedding = mixedEmbedding, pretrainMode = True)
            
                contrastiveLoss = infoNCE(contrast1, contrast2) 

                denoisingCont = 0.0
                for idx, pred in enumerate(cont2):
                    target = view1Cont[:, idx].unsqueeze(1)
                    denoisingCont += F.smooth_l1_loss(pred, target)
                denoisingCont /= idx + 1

                denoisingCat = 0.0
                for idx, pred in enumerate(cat2):
                    target = view1Cat[:, idx]
                    denoisingCat += F.cross_entropy(pred, target) 
                denoisingCat /= idx + 1
                
                
                loss =  contrastiveLoss \
                + denoisingCat * lambdaCat \
                + denoisingCont * lambdaCont \
                + l1Norm * l1 \
                + l2Norm * l2 
                
                valMetrics["Loss"]          += loss
                valMetrics['Contrastive']   += contrastiveLoss
                valMetrics['DenoisingCont'] += denoisingCont
                valMetrics['DenoisingCat']  += denoisingCat
                

                valBar.set_postfix({
                    "Loss": f"{valMetrics['Loss']/(valBar.n+1):.3f}",
                    "Contrastive": f"{valMetrics['Contrastive']/(valBar.n + 1):.3f}",
                    "Cont Denoise": f"{valMetrics['DenoisingCont']/(valBar.n + 1):.3f}",
                    "Cat Denoise": f"{valMetrics['DenoisingCat']/(valBar.n + 1):.3f}"
                })
            valMetrics['Loss']          /= (valBar.n + 1)
            valMetrics['Contrastive']   /= (valBar.n + 1)
            valMetrics['DenoisingCont'] /= (valBar.n + 1)
            valMetrics['DenoisingCat']  /= (valBar.n + 1)
        
            
        scheduler.step()

        currentLr = optimizer.param_groups[0]['lr']
        
        used     = torch.cuda.memory_allocated()  / 2**20
        reserved = torch.cuda.memory_reserved()   / 2**20


        writer.add_scalar("TotalLoss/Train",    trainMetrics["Loss"],         epoch+1)
        writer.add_scalar("TotalLoss/Val",      valMetrics["Loss"],           epoch+1)
        writer.add_scalar("Contrastive/Train",  trainMetrics["Contrastive"],  epoch+1)
        writer.add_scalar("Contrastive/Val",    valMetrics["Contrastive"],    epoch+1)
        writer.add_scalar("DenoisingCont/Train",trainMetrics["DenoisingCont"],epoch+1)
        writer.add_scalar("DenoisingCont/Val",  valMetrics["DenoisingCont"],  epoch+1)
        writer.add_scalar("DenoisingCat/Train", trainMetrics["DenoisingCat"], epoch+1)
        writer.add_scalar("DenoisingCat/Val",   valMetrics["DenoisingCat"],   epoch+1)
        writer.add_scalar("Misc/LearningRate",       currentLr,                    epoch+1)
        writer.add_scalar("Misc/Memory/Allocated",   used,                         epoch+1)
        writer.add_scalar("Misc/Memory/Reserved",    reserved,                     epoch+1)
        writer.flush()

        earlyStop(valMetrics["Loss"], model)
        if earlyStop.early_stop:
            break
