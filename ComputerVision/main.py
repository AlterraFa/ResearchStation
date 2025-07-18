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
    pc_list = load_multi_pc("./dataset/training/truncated_vel", to_idx = 4000)
    processed_data = []
    for _ in tqdm(range(len(pc_list)), desc = "Preprocessing", unit = " Pointclouds"):

        pc = pc_list.pop(0)
        
        pts_xyz     = pc[:, :3].copy()
        reflectance = pc[:, -1].copy()
        n_pts       = pts_xyz.shape[0]

        x_idxs = np.floor((pts_xyz[:,0] - xmin) / resolution).astype(int)
        y_idxs = np.floor((pts_xyz[:,1] - ymin) / resolution).astype(int)
        x_idxs = np.clip(x_idxs, 0, num_x_cells-1)
        y_idxs = np.clip(y_idxs, 0, num_y_cells-1)

        flat_pillar_idx = x_idxs * num_y_cells + y_idxs

        # inverse contains index of each point to its corresponding pillar_idx (recover each point grid via pillar_idx[inv[i]])
        pillar_idx, inv = np.unique(flat_pillar_idx, return_inverse=True) 
        k = pillar_idx.size  # number of non-empty pillars
        

        # Truncate pillar when k > P
        pc_state = truncate_pillar(pillar_index = pillar_idx, pc_mapping = inv,
                                    pts_xyz = pts_xyz, reflectance = reflectance,
                                    tar_pillar_cnt = P)
        
        inv, pts_xyz, reflectance, pillar_idx = pc_state
        # Pad pillar when k < P
        pc_state = pad_pillars(pillar_index = pillar_idx, pc_mapping = inv,
                                pts_xyz = pts_xyz, reflectance = reflectance,
                                num_x = num_x_cells, num_y = num_y_cells,
                                tar_pillar_cnt = P, tar_pc_cnt = N)
        inv, pts_xyz, reflectance, pillar_idx = pc_state
        # Truncate pointcloud when size > N
        pc_state = truncate_pc(pc_mapping = inv, pts_xyz = pts_xyz,
                               reflectance = reflectance, tar_pillar_cnt = P,
                               tar_pc_cnt = N)
        inv, pts_xyz, reflectance = pc_state
            
        
        counts = np.bincount(inv, minlength = P)
        sum_x  = np.bincount(inv, weights = pts_xyz[:,0], minlength = P) # Essentially summation per pillar using inverse mapping
        sum_y  = np.bincount(inv, weights = pts_xyz[:,1], minlength = P) # if the pillar was padded => summation in any axis == 0
        sum_z  = np.bincount(inv, weights = pts_xyz[:,2], minlength = P)
        centroids = np.vstack((sum_x, sum_y, sum_z)).T / counts[:,None]
        

        centroid_per_point = centroids[inv]          # shape (n_pts, 3) redistribute the centroid to each point
        dist_to_centroid   = pts_xyz - centroid_per_point

        ix_unique = pillar_idx // num_y_cells
        iy_unique = pillar_idx %  num_y_cells
        
        pillar_centers = np.vstack((x_pillar_centers[ix_unique],
                                    y_pillar_centers[iy_unique])).T
        pillar_center_per_pt = pillar_centers[inv]
        dist_to_pillar = pts_xyz[:, :2] - pillar_center_per_pt


        # Pad pointcloud when size < N (Note: padding must be behind lifting calculation to avoid shifting centroid)
        pc_state = pad_pc(pc_mapping = inv, pts_xyz = pts_xyz, reflectance = reflectance, 
                          dist_to_centroid = dist_to_centroid, dist_to_pillar = dist_to_pillar,
                          tar_pillar_cnt = P, tar_pc_cnt = N)
        inv, pts_xyz, reflectance, dist_to_centroid, dist_to_pillar = pc_state
        
        pc_9D = np.c_[pts_xyz, dist_to_centroid, dist_to_pillar, reflectance, inv]
        pc_9D = pc_9D.reshape(point_dim + 1, P, N)
