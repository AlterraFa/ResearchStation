import open3d as o3d
import numpy as np
import argparse
import glob
import os, sys
import configparser

from tqdm.auto import tqdm
from utils_3d.math_helper import crop_pc
from utils_3d.disk import load_binary, save_bin

config = configparser.ConfigParser()
config.read("./model/config.ini")

dimension_arg = config["Dimension"]
pillar_arg    = config["Pillarization"]

xmin       = dimension_arg.getfloat("xmin")
xmax       = dimension_arg.getfloat("xmax")
ymin       = dimension_arg.getfloat("ymin")
ymax       = dimension_arg.getfloat("ymax")
zmax       = dimension_arg.getfloat("zmax")
zmin       = dimension_arg.getfloat("zmin")
resolution = dimension_arg.getfloat("resolution")

def truncate(train_folder: str):
    os.makedirs(train_folder + "/truncated_vel", exist_ok = True)

    frame_files = sorted(glob.glob(train_folder + "/velodyne/*"))
    
    for frame_path in tqdm(frame_files, desc = "Trucating", unit = "File"):
        basename = os.path.basename(frame_path)
        name, ext  = os.path.splitext(basename)

        pc = load_binary(frame_path)
        pc = crop_pc(pc, xmax = xmax, xmin = xmin, ymax = ymax, ymin = ymin, zmax = zmax, zmin = zmin)
        save_bin(pc, train_folder + f"/truncated_vel/{name + ext}")

if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="KITTI Truncation"
    )
    
    p.add_argument("train_folder",
        help = "root containing velodyne/, calib/, label_2/")
    args = p.parse_args()
    
    truncate(train_folder = args.train_folder)