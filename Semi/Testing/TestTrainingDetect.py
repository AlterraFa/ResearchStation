import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR, CosineAnnealingWarmRestarts
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.ops import generalized_box_iou as GIoU, box_convert as bboxConvert
from scipy.optimize import linear_sum_assignment
import xml.etree.ElementTree as ET

from utils.imageTransformer import DeTr
from utils.helper import EarlyStopping
from tensorboardX import SummaryWriter

from tqdm.auto import tqdm
from typing import List

normTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class DetectionDataset(Dataset):
    def __init__(self, dataset, className: List[str], imgSize = 640):
        super().__init__()
        
        self.dataset = dataset
        self.imgSize = imgSize
        self.CLASSES = className
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, target = self.dataset[index]
        ann = target['annotation']
        objs = ann['object']
        if isinstance(objs, dict):
            objs = [objs]

        boxes = torch.tensor([
            [
            float(o['bndbox']['xmin']),
            float(o['bndbox']['xmax']),
            float(o['bndbox']['ymin']),
            float(o['bndbox']['ymax'])
            ]
        for o in objs], dtype = torch.float32)
        labels = torch.tensor([self.CLASSES.index(o['name']) for o in objs], dtype = torch.long)
        
        H, W = img.shape[1:]
        scale = min(self.imgSize / H, self.imgSize / W)
        newH, newW = int(H * scale), int(W * scale)
        img = F.resize(img, (newH, newW))
        padH, padW = self.imgSize - newH, self.imgSize - newW
        left = padW // 2
        right = padW - left
        top = padH // 2
        bottom = padH - top
        img = F.pad(img, (left, top, right, bottom), fill = 0)
        
        boxes = boxes.clone().float()
        boxes *= scale
        boxes[:, [0, 1]] += left
        boxes[:, [2, 3]] += top
        boxes /= self.imgSize
        
        return img, (labels, boxes)
    
    def collate_fn(self, batch):
        imgs      = [item[0] for item in batch]
        labels    = [item[1][0] for item in batch]
        boxes     = [item[1][1] for item in batch]

        imgs = torch.stack(imgs, dim=0)
        return imgs, labels, boxes
        
def fastUniSplit(
    voc_root: str,
    image_set: str,
    classes: List[str],
    val_fraction: float=0.2,
    seed: int=42
):
    import numpy as np
    from sklearn.model_selection import train_test_split

    # 1) load the list of ids in this split
    splits_dir = os.path.join(voc_root, "ImageSets", "Main")
    with open(os.path.join(splits_dir, f"{image_set}.txt")) as f:
        ids = [line.strip() for line in f]
    N = len(ids)

    # 2) parse each XML and grab the first object’s class
    ann_dir = os.path.join(voc_root, "Annotations")
    primary = []
    for idx in ids:
        xml_path = os.path.join(ann_dir, f"{idx}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objs = root.findall("object")
        if objs:
            name = objs[0].find("name").text
            primary.append(classes.index(name))
        else:
            primary.append(-1)

    primary = np.array(primary)
    all_idxs = np.arange(N)
    train_idxs, val_idxs = train_test_split(
        all_idxs,
        test_size=val_fraction,
        random_state=seed,
        stratify=primary
    )
    return train_idxs.tolist(), val_idxs.tolist()

class Writer(SummaryWriter):
    def __init__(self, log_dir: str = "./"):
        super().__init__()
        self.log_dir = log_dir
        self.log_dir = os.path.abspath(self.log_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def add_graph(self, model, input_to_model, verbose=False):
        super().add_graph(model, input_to_model, verbose=verbose)
        self.flush()
        
if __name__ == "__main__":
    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    
    vocRoot = "./datasets/VOCdevkit/VOC2012"
    CLASS = [
        'aeroplane','bicycle','bird','boat','bottle',
        'bus','car','cat','chair','cow','diningtable',
        'dog','horse','motorbike','person','pottedplant',
        'sheep','sofa','train','tvmonitor'
    ]

    trainIdx, valIdx = fastUniSplit(vocRoot, "trainval", CLASS, val_fraction = 0.1)
    
    full = datasets.VOCDetection(
        root="./datasets",
        year="2012",
        image_set="trainval",
        download = False,    # will download and extract the tar for you
        transform = normTransform     # any torchvision transforms you want
    )

    fullDs  = DetectionDataset(full, CLASS)
    trainDs = Subset(fullDs, trainIdx)
    valDs   = Subset(fullDs, valIdx)
    
    trainLoader = DataLoader(trainDs, batch_size = 7, shuffle = True, num_workers = 12, persistent_workers = True, pin_memory = True, collate_fn = fullDs.collate_fn)
    valLoader   = DataLoader(valDs, batch_size = 3, shuffle = True, num_workers = 12, persistent_workers = True, pin_memory = True, collate_fn = fullDs.collate_fn)

    model = DeTr(numClasses = len(CLASS),
                 nEncoders = 6,
                 nDecoders = 6).to(gpu, non_blocking=True)
    
    data = next(iter(trainLoader))[0]
    data = data.to(gpu, non_blocking=True)
    writer = Writer(log_dir = "./runs/DeTr")
    writer.add_graph(model, data)
    writer.flush()
    

    epochs = 400; switchEpoch = 60; initLR = 7.5e-4; finalLR = 1e-10; l1 = 1e-5; l2 = 1e-5
    αGIoUBox = 1; αL1Box = 1; αClass = 1
    weights = torch.ones(len(CLASS) + 1, device = gpu); weights[len(CLASS)] = 0.1
    classCriterion  = nn.CrossEntropyLoss(label_smoothing = .1, reduction = "none", weight = weights)
    bboxL1Criterion = nn.SmoothL1Loss(reduction = "none")
    optimizer       = optim.AdamW(model.parameters(), lr = initLR, betas = (0.9, 0.999))
    scheduler       = CosineAnnealingLR(optimizer = optimizer, T_max = epochs, eta_min = finalLR)
    earlyStop       = EarlyStopping(patience = 50, path = f"./Detr.pt", verbose = True)
    
    pbar = tqdm(range(epochs), desc="Training Epochs", position = 0)
    for epoch in pbar:
        model.train()
        
        trainBar = tqdm(trainLoader, desc = "Train", position = 1, leave = False); trainCnt = 0
        trainMetrics = {"Total": 0, "Supervised": 0, "Consistency": 0, "Accuracy": 0}
        for trainCnt, (xBatch, labels, bbox) in enumerate(trainBar):
            optimizer.zero_grad()
            xBatch = xBatch.to(gpu, non_blocking = True)
            batchSize = xBatch.shape[0]


            classLogits, bboxProposal = model(xBatch)
            # Since the hungarian algorithm works with 2D tensors, we need to loop through the batch
            # Or I need to write custom hungarian algorithm for 3D tensors
            
            loss = 0.0
            for i in range(batchSize):
                numDetections = labels[i].shape[0]
                if numDetections == 0:
                    classTargets = torch.full((model.proposalSize, ), len(CLASS), device = gpu, dtype = torch.long)
                    classLoss    = classCriterion(classLogits[i], classTargets).mean()
                    loss         = αClass * classLoss
                    continue
                classGT = labels[i].to(gpu)
                bboxGT  = bbox[i].to(gpu) 

                # This is still missing background loss padding (Will work on it later)
                labelsFlat        = classGT.unsqueeze(0).expand(model.proposalSize, -1).reshape(-1)
                classLogitsFlat   = classLogits[i].unsqueeze(1).expand(-1, numDetections, -1).reshape(-1, len(CLASS) + 1)
                hungarianClassMat = classCriterion(classLogitsFlat, labelsFlat).reshape(model.proposalSize, -1)

                bboxConverted        = bboxConvert(bboxProposal[i], in_fmt = "xywh", out_fmt = "xyxy") # Need the output to be xywh since shit won't work if it is raw xyxy and creates degenerate bb (negative area)
                bboxPredExpandedFlat = bboxConverted.unsqueeze(1).expand(-1, numDetections, -1).reshape(-1, 4)
                bboxGTExpandedFlat   = bboxGT.unsqueeze(0).expand(model.proposalSize, -1, -1).reshape(-1, 4)
                bboxL1Mat            = bboxL1Criterion(bboxPredExpandedFlat, bboxGTExpandedFlat).reshape(model.proposalSize, numDetections, -1).mean(dim = -1)
                bboxGIoUMat          = GIoU(bboxConverted, bboxGT)

                hungarianCost = αClass * hungarianClassMat + αL1Box * bboxL1Mat + αGIoUBox * - bboxGIoUMat
                hungarianCost = hungarianCost.cpu().detach().numpy()

                rowIdx, colIdx = linear_sum_assignment(hungarianCost)
                
                classTargets = torch.full((model.proposalSize, ), len(CLASS), device = gpu, dtype = torch.long)
                classTargets[rowIdx] = classGT[colIdx]
                classLoss = classCriterion(classLogits[i], classTargets).mean()

                
                bboxL1Loss = bboxL1Criterion(bboxConverted[rowIdx], bboxGT[colIdx]).mean()
                bboxGIoULoss = (1.0 - bboxGIoUMat[rowIdx, colIdx]).mean()
        
                loss += αClass * classLoss + αL1Box * bboxL1Loss + αGIoUBox * bboxGIoULoss        
            
            loss /= batchSize
            loss.backward()

            # supervisedLoss = supervisedCriterion(logits, yBatch).mean()
            # distribution   = torch.softmax(logits, dim = 1)
            # correct        = (torch.argmax(distribution, dim = 1) == yBatch).sum(); trainCnt += yBatch.shape[0]

            # with torch.no_grad():
            #     unlabeledWeak      = unlabeledWeak.to(device)
            #     wLogits            = model(unlabeledWeak.to(device))
            #     qWeak              = torch.softmax(wLogits, dim = 1)
            #     confs, pseudoLabel = qWeak.max(dim = 1)
            #     # pseudoLabel        = pseudoLabel.detach()
            #     pseudoLabel        = alignment(qWeak)
            #     mask               = (confs >= tau).float()

            
            # consecLoss = 0 # Augmentation anchoring
            # for unlabeledStrong in unlabeledStrongList:
            #     unlabeledStrong = unlabeledStrong.to(device, non_blocking = True)

            #     sLogits      = model(unlabeledStrong)
            #     scalarLoss   = (mask * unsupervisedCriterion(sLogits, pseudoLabel)).mean()
            #     consecLoss  += scalarLoss

                
            # consistencyLoss = consecLoss / reaugmentApply

            # weightParams = [p for n, p in model.named_parameters()
            #                 if p.requires_grad and "weight" in n]
            # l1Norm = sum(p.abs().sum() for p in weightParams)
            # l2Norm = sum(p.pow(2.0).sum() for p in weightParams)
            
            # loss = supervisedLoss \
            #         + consistencyLoss \
            #         + l1Norm * l1 \
            #         + l2Norm * l2
            # loss.backward()
            # optimizer.step()


            # trainMetrics["Consistency"] += consistencyLoss.item()
            # trainMetrics["Total"]       += loss.item()
            # trainMetrics["Supervised"]  += supervisedLoss.item()
            # trainMetrics["Accuracy"]    += correct.item()
            
            trainBar.set_postfix({
                "T": f"{trainMetrics['Total']/ (trainBar.n+1):.3f}",
                "S": f"{trainMetrics['Supervised']/(trainBar.n+1):.3f}",
                "C": f"{trainMetrics['Consistency']/(trainBar.n+1):.3f}",
                "Acc": f"{trainMetrics['Accuracy']/(trainBar.n+1) * 100:.2f}%",
            })
        
        trainMetrics["Consistency"] /= len(trainLoader)
        trainMetrics["Total"]       /= len(trainLoader)
        trainMetrics["Supervised"]  /= len(trainLoader)
        trainMetrics["Accuracy"]    /= len(trainDs)

        with torch.no_grad():
            valBar = tqdm(valLoader, desc = "Val", position = 2, leave = False)
            valMetrics = {"Accuracy": 0, "Cost": 0}; valCnt = 0
            for (xBatch, yBatch) in valBar:
                xBatch = xBatch.to(gpu)
                yBatch = yBatch.to(gpu)

                supervisedLogits = model(xBatch)
                # supervisedLoss   = supervisedCriterion(supervisedLogits, yBatch)
                # supervisedDist   = torch.softmax(supervisedLogits, dim = 1) 

                # valMetrics['Accuracy'] += (torch.argmax(supervisedDist, dim = 1) == yBatch).sum().item()
                # valMetrics["Cost"]     += supervisedLoss.item()
                
                # valCnt += yBatch.shape[0]
                
                # valBar.set_postfix({
                #     "Acc": f"{valMetrics['Accuracy'] / (valCnt) * 100:.3f}%",
                #     "Cost": f"{valMetrics['Cost'] / (valBar.n+1):.3f}"
                # })

            valMetrics['Cost']     /= len(valLoader)
            valMetrics['Accuracy'] /= len(valDs)


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
            f"No update: {earlyStop.counter}/{earlyStop.patience}"
        )
        # writer.add_scalar("Loss/Supervised",     trainMetrics["Supervised"], epoch+1)
        # writer.add_scalar("Loss/Consistency",    trainMetrics["Consistency"], epoch+1)
        # writer.add_scalar("Loss/Total",          trainMetrics["Total"],       epoch+1)
        # writer.add_scalar("Accuracy/Train",      100*trainMetrics["Accuracy"],epoch+1)
        # writer.add_scalar("Loss/Validation",     valMetrics["Cost"],          epoch+1)
        # writer.add_scalar("Accuracy/Validation", 100*valMetrics["Accuracy"],  epoch+1)
        # writer.add_scalar("Misc/LearningRate",   currentLr,                    epoch+1)
        # writer.flush()
        
        earlyStop(valMetrics['Cost'], model)
        if earlyStop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break