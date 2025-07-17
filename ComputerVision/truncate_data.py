import open3d as o3d
import numpy as np
import argparse
import glob
import os, sys

from tqdm.auto import tqdm
from utils_3d.math_helper import crop_pc
from utils_3d.loader import load_bin, save_bin

def truncate(train_folder: str):
    os.makedirs(train_folder + "/truncated_vel", exist_ok = True)

    frame_files = sorted(glob.glob(train_folder + "/velodyne/*"))
    
    for frame_path in tqdm(frame_files, desc = "Trucating", unit = "File"):
        basename = os.path.basename(frame_path)
        name, ext  = os.path.splitext(basename)

        pc = load_bin(frame_path)
        pc = crop_pc(pc)
        save_bin(pc, train_folder + f"/truncated_vel/{name + ext}")

if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="KITTI Truncation"
    )
    
    p.add_argument("train_folder",
        help = "root containing velodyne/, calib/, label_2/")
    args = p.parse_args()
    
    truncate(train_folder = args.train_folder)