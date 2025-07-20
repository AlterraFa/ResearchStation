import os
import numpy as np
import configparser
import argparse
import torch

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from utils_3d.disk import load_multi_pc
from utils_3d.preprocessing import PillarDataset

config = configparser.ConfigParser()
config.read("./model/config.ini")

dimension_arg = config["Dimension"]
pillar_arg    = config["Pillarization"]
dataset_arg   = config["Dataset"]

xmin       = dimension_arg.getfloat("xmin")
xmax       = dimension_arg.getfloat("xmax")
ymin       = dimension_arg.getfloat("ymin")
ymax       = dimension_arg.getfloat("ymax")
resolution = dimension_arg.getfloat("resolution")

P = pillar_arg.getint("P")
N = pillar_arg.getint("N")

batch_size  = dataset_arg.getint("batch_size")
num_workers = dataset_arg.getint("num_workers")
persistent  = dataset_arg.getboolean("persistent")
shuffle     = dataset_arg.getboolean("shuffle")

PILLARARGS = {
    "xmin": xmin, "xmax": xmax,
    "ymin": ymin, "ymax": ymax,
    "resolution": resolution,
    "num_pillars": P,
    "num_pc": N
}


def main(path: str):
    np.random.seed(12)
    os.makedirs(path + "/naquium", exist_ok = True)
    
    pc_list = load_multi_pc("./dataset/training/truncated_vel", to_idx = 5000)
    pillar_ds = PillarDataset(pointclouds = pc_list, **PILLARARGS)
    pillar_loader = DataLoader(pillar_ds, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, persistent_workers = persistent)
    del pillar_ds, pc_list

    for data in tqdm(pillar_loader, desc = "Test loading", unit = " pointclouds", unit_scale = batch_size):
        ...
    

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="KITTI Truncation"
    )
    
    p.add_argument("train_folder",
        help = "root containing velodyne/, calib/, label_2/")
    args = p.parse_args()
    
    main(path = args.train_folder)
