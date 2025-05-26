import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

import torch
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET
from torchvision import datasets, transforms
from torchvision.ops import box_iou as IoU, box_convert as bboxConvert
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F


from utils.imageTransformer import DeTr
from utils.helper import EarlyStopping, DetectionDataset
from tensorboardX import SummaryWriter

from tqdm.auto import tqdm
from typing import List

normTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

        
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

def detectionEval(predBoxes: torch.Tensor,
                scoreLogits: torch.Tensor,
                gtBoxes: torch.Tensor,
                gtLabels: torch.Tensor,
                iouThreshold: float = 0.5,
                scoreThreshold: float = 0.5) -> float:
    device = predBoxes.device
    P, numClasses = scoreLogits.shape

    probs    = F.softmax(scoreLogits, dim=1)
    bgIdx    = numClasses - 1
    objConfs = 1.0 - probs[:, bgIdx]

    keep = objConfs >= scoreThreshold
    if keep.sum() == 0:
        return 0.0, 0.0, gtBoxes.size(0)
    boxes      = bboxConvert(predBoxes[keep], in_fmt="cxcywh", out_fmt="xyxy")
    scores     = objConfs[keep]
    predLabels = probs[keep, :bgIdx].argmax(dim=1)

    scores, order = scores.sort(descending=True)
    boxes         = boxes[order]
    predLabels    = predLabels[order]

    G = gtBoxes.size(0)
    if G == 0:
        return 0.0, boxes.size(0), 0.0

    iouMat    = IoU(boxes, gtBoxes)
    matchedGT = torch.zeros(G, dtype=torch.bool, device=device)

    tp = 0
    fp = 0

    for i in range(boxes.size(0)):
        bestIoU, bestIdx = iouMat[i].max(dim=0)
        if bestIoU >= iouThreshold             \
           and not matchedGT[bestIdx]           \
           and predLabels[i] == gtLabels[bestIdx].to(device):
            tp += 1
            matchedGT[bestIdx] = True
        else:
            fp += 1
            
    fn = (~matchedGT).sum().item()

    return tp, fp, fn
        
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
    writer = SummaryWriter()
    writer.add_graph(model, data)
    writer.flush()
    

    epochs = 400; switchEpoch = 60; initLR = 7.5e-4; finalLR = 1e-10; l1 = 1e-5; l2 = 1e-5
    alphaGIoUBox = 2.0; alphaL1Box = 1.0; alphaClass = 1.5
    weights = torch.ones(len(CLASS) + 1, device = gpu); weights[len(CLASS)] = 0.1
    auxTraining = True  
    
    classCriterion  = nn.CrossEntropyLoss(label_smoothing = .1, reduction = "none", weight = weights)
    bboxL1Criterion = nn.SmoothL1Loss(reduction = "none")
    optimizer       = optim.AdamW(model.parameters(), lr = initLR, betas = (0.9, 0.999))
    scheduler       = CosineAnnealingLR(optimizer = optimizer, T_max = epochs, eta_min = finalLR)
    earlyStop       = EarlyStopping(patience = 50, path = f"./Detr.pt", verbose = True)
    
    pbar = tqdm(range(epochs), desc="Training Epochs", position = 0)
    for epoch in pbar:
        model.train()
        
        trainBar = tqdm(trainLoader, desc = "Train", position = 1, leave = False); trainCnt = 0
        trainMetrics = {"Total": 0, "Supervised": 0, "Precision": 0, "Recall": 0, "F1": 0}
        for xBatch, labels, bbox in trainBar:
            optimizer.zero_grad()
            xBatch = xBatch.to(gpu, non_blocking = True)
            batchSize = xBatch.shape[0]


            classLogits, bboxProposal = model(xBatch, auxiliary = auxTraining)
            

            supervisedLoss = model.loss(classLogits = classLogits, labels = labels,
                                        bboxProposal = bboxProposal, bbox = bbox,
                                        classCriterion = classCriterion, boxCriterion = bboxL1Criterion,
                                        alphaClass = alphaClass, alphaL1Box = alphaL1Box, alphaGIoUBox = alphaGIoUBox)
            
            weightParams = [p for n, p in model.named_parameters()
                            if p.requires_grad and "weight" in n]
            l1Norm = sum(p.abs().sum() for p in weightParams)
            l2Norm = sum(p.pow(2.0).sum() for p in weightParams)
            
            loss = supervisedLoss \
                    + l1Norm * l1 \
                    + l2Norm * l2
            loss.backward()
            optimizer.step()

            tp, fn, fp = 0, 0, 0
            for i in range(batchSize):
                tp_, fp_, fn_ = detectionEval(
                    predBoxes = bboxProposal[i][-1] if auxTraining else bboxProposal[i],
                    scoreLogits = classLogits[i][-1] if auxTraining else classLogits[i],
                    gtBoxes = bbox[i].to(gpu),
                    gtLabels = labels[i].to(gpu),
                    iouThreshold = 0.3,
                    scoreThreshold = 0.3
                )
                tp += tp_   
                fn += fn_
                fp += fp_

        

            trainMetrics["Precision"] += tp / (tp + fp + 1e-8)
            trainMetrics["Recall"] += tp / (tp + fn + 1e-8)
            trainMetrics["F1"] = 2 * trainMetrics["Precision"] * trainMetrics["Recall"] \
                                  / (trainMetrics["Precision"] + trainMetrics["Recall"] + 1e-8)
            
            trainMetrics["Total"]       += loss.item()
            trainMetrics["Supervised"]  += supervisedLoss.item()
            
            trainBar.set_postfix({
                "Total": f"{trainMetrics['Total']/ (trainBar.n+1):.3f}",
                "Supervised": f"{trainMetrics['Supervised']/(trainBar.n+1):.3f}",
                "Prec": f"{trainMetrics['Precision']:.3f}",
                "Recall": f"{trainMetrics['Recall']:.3f}",
                "F1": f"{trainMetrics['F1']:.3f}",
            })
        
        trainMetrics["Total"]      /= len(trainLoader)
        trainMetrics["Supervised"] /= len(trainLoader)

        with torch.no_grad():
            valBar = tqdm(valLoader, desc = "Val", position = 2, leave = False)
            valMetrics = {"Supervised": 0, "Precision": 0, "Recall": 0, "F1": 0}; valCnt = 0
            for xBatch, labels, bbox in valBar:
                xBatch = xBatch.to(gpu, non_blocking = True)
                batchSize = xBatch.shape[0]

                classLogits, bboxProposal = model(xBatch, auxiliary = auxTraining)

                supervisedLoss = model.loss(classLogits = classLogits, labels = labels,
                                            bboxProposal = bboxProposal, bbox = bbox,
                                            classCriterion = classCriterion, boxCriterion = bboxL1Criterion,
                                            alphaClass = alphaClass, alphaL1Box = alphaL1Box, alphaGIoUBox = alphaGIoUBox)

                tp, fn, fp = 0, 0, 0
                for i in range(batchSize):
                    tp_, fn_, fp_ = detectionEval(
                        predBoxes = bboxProposal[i][-1] if auxTraining else bboxProposal[i],
                        scoreLogits = classLogits[i][-1] if auxTraining else classLogits[i],
                        gtBoxes = bbox[i].to(gpu),
                        gtLabels = labels[i].to(gpu),
                        iouThreshold = 0.3,
                        scoreThreshold = 0.3
                    )
                    tp += tp_   
                    fn += fn_
                    fp += fp_


                valMetrics["Supervised"] += supervisedLoss.item()
                valMetrics["Precision"] += tp / (tp + fp + 1e-8)
                valMetrics["Recall"] += tp / (tp + fn + 1e-8)
                valMetrics["F1"] = 2 * valMetrics["Precision"] * valMetrics["Recall"] \
                                    / (valMetrics["Precision"] + valMetrics["Recall"] + 1e-8)
                
                valBar.set_postfix({
                    "Supervised": f"{valMetrics['Supervised'] / (valBar.n+1):.3f}",
                    "Prec": f"{valMetrics['Precision'] / (valBar.n + 1):.3f}",
                    "Recall": f"{valMetrics['Recall'] / (valBar.n + 1):.3f}",
                    "F1": f"{valMetrics['F1'] / (valBar.n + 1):.3f}",
                })

            valMetrics['Supervised'] /= len(valLoader)


        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']
        
        used     = torch.cuda.memory_allocated()  / 2**20
        reserved = torch.cuda.memory_reserved()   / 2**20

        
        tqdm.write(
            f"Epoch {epoch+1}/{epochs} — "
            f"Sup Train: {trainMetrics['Supervised']:.4f}, ",
            f"Prec Train: {trainMetrics['Precision']:.4f}, ",
            f"Rec Train: {trainMetrics['Recall']:.4f}, ",
            f"F1 Train: {trainMetrics['F1']:.4f}, ",
            f"Sup Val: {valMetrics['Supervised']:.4f}, ",
            f"Prec Val: {valMetrics['Precision']:.4f}, ",
            f"Rec Val: {valMetrics['Recall']:.4f}, ",
            f"F1 Val: {valMetrics['F1']:.4f}, ",
            f"No update: {earlyStop.counter}/{earlyStop.patience}"
        )
        writer.add_scalar("Loss/Supervised Train",   trainMetrics["Supervised"], epoch+1)
        writer.add_scalar("Loss/Supervised Val",     valMetrics["Supervised"],   epoch+1)
        writer.add_scalar("Metrics/Train/Precision", trainMetrics["Precision"],  epoch+1)
        writer.add_scalar("Metrics/Train/Recall",    trainMetrics["Recall"],     epoch+1)
        writer.add_scalar("Metrics/Train/F1",        trainMetrics["F1"],         epoch+1)
        writer.add_scalar("Metrics/Val/Precision",   valMetrics["Precision"],   epoch+1)
        writer.add_scalar("Metrics/Val/Recall",      valMetrics["Recall"],      epoch+1)
        writer.add_scalar("Metrics/Val/F1",          valMetrics["F1"],          epoch+1)
        writer.add_scalar("Misc/LearningRate",       currentLr,                    epoch+1)
        writer.flush()
        
        earlyStop(valMetrics['Cost'], model)
        if earlyStop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break