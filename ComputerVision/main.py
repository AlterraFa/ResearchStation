import numpy as np

from collections import defaultdict
from tqdm.auto import tqdm
from model.Point_Pillar import *
from utils_3d.loader import load_multi_pc
from utils_3d.preprocessing import *

xmax, xmin = 70.4, 0 # meter
ymax, ymin = 40, -40 # meter
zmax, zmin = -3, 1 # meter
resolution = 0.16 # m/pixel
P = 10000
N = 75
point_dim = 9

    
if __name__ == "__main__":
    
    np.random.seed(12)
    
    width  = xmax - xmin
    length = ymax - ymin
    
    num_x_cells = int(np.ceil(length / resolution))
    num_y_cells = int(np.ceil(width / resolution))
    total_pillars = num_x_cells * num_y_cells
    
    x_pillar_centers = xmin + (np.arange(num_x_cells) + 0.5) * resolution
    y_pillar_centers = ymin + (np.arange(num_y_cells) + 0.5) * resolution
    
    print(f"Pseudo image resolution: ({num_x_cells}, {num_y_cells})")
    pc_list = load_multi_pc("./dataset/training/truncated_vel", to_idx = 1000)
    
    pillar = Pillarization(xmax, xmin, ymax, ymin, resolution, P, N)
    processed_data = []
    for _ in tqdm(range(len(pc_list)), desc = "Preprocessing", unit = " Pointclouds"):

        pc = pc_list.pop(0)

        pc_9D, pillar_index, pc_mapping = pillar.apply(pc_data = pc)
