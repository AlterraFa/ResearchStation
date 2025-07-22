import os
import numpy as np
import configparser
import argparse
import torch
import math

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from utils_3d.disk import load_multi, load_binary, load_calib, load_labels
from utils_3d.preprocessing import PillarDataset
from model.Point_Pillar import PointPillar

config = configparser.ConfigParser()
config.read("./model/config.ini")

dimension_arg = config["Dimension"]
pillar_arg    = config["Pillarization"]
dataset_arg   = config["Dataset"]
model_arg     = config["Model"]

xmin       = dimension_arg.getfloat("xmin")
xmax       = dimension_arg.getfloat("xmax")
ymin       = dimension_arg.getfloat("ymin")
ymax       = dimension_arg.getfloat("ymax")
resolution = dimension_arg.getfloat("resolution")
H, W = math.ceil((xmax - xmin) / resolution), math.ceil((ymax - ymin) / resolution)

P = pillar_arg.getint("P")
N = pillar_arg.getint("N")

batch_size      = dataset_arg.getint("batch_size")
num_workers     = dataset_arg.getint("num_workers")
persistent      = dataset_arg.getboolean("persistent")
shuffle         = dataset_arg.getboolean("shuffle")
allowed_classes = dataset_arg.get("allowed_classes").split(", ")

point_dim    = model_arg.getint("point_dim")
target_dim   = model_arg.getint("expand_dim")
loc_features = model_arg.getint("loc_features")
num_classes  = model_arg.getint("loc_features")

PILLARARGS = {
    "xmin": xmin, "xmax": xmax,
    "ymin": ymin, "ymax": ymax,
    "resolution": resolution,
    "num_pillars": P,
    "num_pc": N
}


np.random.seed(12)
torch.manual_seed(12)
device = torch.device('cuda')

def main(path: str):
    
    start_idx, end_idx = 0, 500
    pc_list    = load_multi(path + "/truncated_vel", load_binary, from_idx = start_idx, to_idx = end_idx)
    label_list = load_multi(path + "/label_2", load_labels, from_idx = start_idx, to_idx = end_idx)
    calib_list = load_multi(path + "/calib", load_calib, from_idx = start_idx, to_idx = end_idx)

    label_list_filtered = []
    for label in label_list:
        label_filtered = []
        for obj in label:
            obj['class'] = obj['class'].lower()
            if obj['class'] in allowed_classes:
                label_filtered.append(obj)
        label_list_filtered.append(label_filtered)
    label_list = label_list_filtered
    
    pillar_ds = PillarDataset(pointclouds = pc_list, 
                              labels = label_list, 
                              calibs = calib_list,
                              allowed = allowed_classes, 
                              **PILLARARGS)
    pillar_loader = DataLoader(pillar_ds, 
                               batch_size = batch_size, 
                               shuffle = shuffle, 
                               num_workers = num_workers, 
                               persistent_workers = persistent, 
                               collate_fn = pillar_ds.collate_fn)
    
    del pillar_ds, pc_list
    
    
    model = PointPillar(num_classes = num_classes, 
                        loc_features = loc_features, 
                        point_dim = point_dim, 
                        target_dim = target_dim, 
                        height = H, width = W)
    model.to(device)

    for data in tqdm(pillar_loader, desc = "Test loading", unit = " pointclouds", unit_scale = batch_size):
        pointcloud   = data["features"].to(device)
        pillar_index = data["pillar_index"].to(device)

        locs, confs = model(pointcloud, pillar_index)
    

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="KITTI Truncation"
    )
    
    p.add_argument("train_folder",
        help = "root containing velodyne/, calib/, label_2/")
    args = p.parse_args()
    
    main(path = args.train_folder)
