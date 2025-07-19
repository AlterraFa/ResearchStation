import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _truncate_pc_numba(pc_mapping, pts_xyz, reflectance,
                       target_pillar, target_pc):
    n_pts = pc_mapping.shape[0]
    keep_mask = np.ones(n_pts, np.bool_)

    # 1) count per pillar
    counts = np.bincount(pc_mapping, minlength=target_pillar)

    # 2) for each pillar in parallel
    for pid in prange(target_pillar):
        cnt = counts[pid]
        if cnt > target_pc:
            # collect indices of that pillar
            idxs = np.empty(cnt, np.int64)
            c = 0
            for i in range(n_pts):
                if pc_mapping[i] == pid:
                    idxs[c] = i
                    c += 1

            # pick exactly (cnt - target_pc) to drop
            to_drop = np.random.choice(cnt, cnt - target_pc, replace=False)
            for j in to_drop:
                keep_mask[idxs[j]] = False

    # 3) apply mask
    return (
        pc_mapping[keep_mask],
        pts_xyz[keep_mask],
        reflectance[keep_mask]
    )

class Pillarization():
    def __init__(self, xmax: float, xmin: float, ymax: float, ymin: float, resolution: float, P: int, N: int):
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.resolution    = resolution
        self.target_pillar = P
        self.target_pc     = N
        self.pc_dim        = 9
        
        width  = xmax - xmin
        length = ymax - ymin
        
        self.num_x_cells = int(np.ceil(width / resolution))
        self.num_y_cells = int(np.ceil(length / resolution))
        self.total_pillars = self.num_x_cells * self.num_y_cells

        print(f"Pseudo image resolution: ({self.num_x_cells}, {self.num_y_cells})")
        
        self.x_centers = xmin + (np.arange(self.num_x_cells) + 0.5) * resolution
        self.y_centers = ymin + (np.arange(self.num_y_cells) + 0.5) * resolution

    pillar_encoder = lambda self, x_idx, y_idx: x_idx * self.num_y_cells + y_idx
    pillar_decoder = lambda self, pillar_idx: (pillar_idx // self.num_y_cells, pillar_idx % self.num_y_cells)


    def apply(self, pc_data: np.ndarray): 

        pts_xyz     = pc_data[:, :3].copy()
        reflectance = pc_data[:, -1].copy()

        x_idxs = np.floor((pts_xyz[:,0] - self.xmin) / self.resolution).astype(int)
        y_idxs = np.floor((pts_xyz[:,1] - self.ymin) / self.resolution).astype(int)
        x_idxs = np.clip(x_idxs, 0, self.num_x_cells-1)
        y_idxs = np.clip(y_idxs, 0, self.num_y_cells-1)

        flat_pillar_idx = self.pillar_encoder(x_idxs, y_idxs)

        # inverse contains index of each point to its corresponding pillar_idx (recover each point grid via pillar_idx[inv[i]])
        pillar_idx, inv = np.unique(flat_pillar_idx, return_inverse=True) 

        # Truncate pillar when k > P
        pc_state = self.truncate_pillar(pillar_index = pillar_idx, 
                                        pc_mapping = inv, 
                                        pts_xyz = pts_xyz, 
                                        reflectance = reflectance)
        inv, pts_xyz, reflectance, pillar_idx = pc_state

        # Pad pillar when k < P
        pc_state = self.pad_pillars(pillar_index = pillar_idx, 
                                    pc_mapping = inv, 
                                    pts_xyz = pts_xyz, 
                                    reflectance = reflectance)
        inv, pts_xyz, reflectance, pillar_idx = pc_state

        # Truncate pointcloud when size > N
        pc_state = self.truncate_pc(pc_mapping = inv, 
                                    pts_xyz = pts_xyz, 
                                    reflectance = reflectance)
        inv, pts_xyz, reflectance = pc_state
            
        # Calculate distance of each point to its arithmetic mean, distance to pillar center        
        dist_to_centroid, dist_to_pillar = self.pillar_stats(pts_xyz = pts_xyz, 
                                                             pillar_index = pillar_idx, 
                                                             pc_mapping = inv)

        # Pad pointcloud when size < N (Note: padding must be behind lifting calculation to avoid shifting centroid)
        pc_state = self.fast_pad_pc(pc_mapping = inv, 
                                    pts_xyz = pts_xyz, 
                                    reflectance = reflectance, 
                                    dist_to_centroid = dist_to_centroid, 
                                    dist_to_pillar = dist_to_pillar)
        inv, pts_xyz, reflectance, dist_to_centroid, dist_to_pillar = pc_state

        
        pc_9D = np.c_[pts_xyz, dist_to_centroid, dist_to_pillar, reflectance]
        pc_9D = pc_9D.reshape(self.pc_dim, self.target_pillar, self.target_pc)
        return pc_9D, pillar_idx, inv

    def truncate_pillar(self,
                        pillar_index: np.ndarray, pc_mapping: np.ndarray,
                        pts_xyz: np.ndarray, reflectance: np.ndarray):

        curr_pillar_cnt = pillar_index.shape[0]
        if curr_pillar_cnt <= self.target_pillar:
            return pc_mapping, pts_xyz, reflectance, pillar_index
        
        keep_index = np.random.choice(curr_pillar_cnt, size = self.target_pillar, replace = False)
        
        remap = -np.ones(curr_pillar_cnt, dtype=int)
        remap[keep_index] = np.arange(self.target_pillar)
        new_inv = remap[pc_mapping]

        keep_pts_mask = new_inv >= 0
        
        return (
            new_inv[keep_pts_mask],
            pts_xyz[keep_pts_mask],        
            reflectance[keep_pts_mask],
            pillar_index[keep_index]
        )

    def pad_pillars(self, 
                    pillar_index: np.ndarray,
                    pc_mapping:   np.ndarray,
                    pts_xyz:      np.ndarray,
                    reflectance:  np.ndarray):
        curr_cnt = pillar_index.shape[0]
        if curr_cnt >= self.target_pillar:
            return pc_mapping, pts_xyz, reflectance, pillar_index

        occ = np.zeros((self.num_x_cells, self.num_y_cells), dtype=bool)
        ix_u, iy_u = self.pillar_decoder(pillar_index)
        occ[ix_u, iy_u] = True
        empty_coords = np.argwhere(~occ)  

        pad_cnt = self.target_pillar - curr_cnt
        sel = empty_coords[np.random.choice(
            empty_coords.shape[0],
            size=pad_cnt,
            replace=False
        )]
        pad_globals = self.pillar_encoder(sel[:,0], sel[:,1])

        new_slots  = np.arange(curr_cnt, curr_cnt + pad_cnt)
        dummy_inv  = np.repeat(new_slots, self.target_pc)
        dummy_xyz  = np.zeros((pad_cnt * self.target_pc, 3))
        dummy_ref  = np.zeros((pad_cnt * self.target_pc,))

        return (
            np.concatenate([pc_mapping,   dummy_inv]),
            np.vstack([pts_xyz,           dummy_xyz]),
            np.concatenate([reflectance,  dummy_ref]),
            np.concatenate([pillar_index, pad_globals])                        
        )
        
    def truncate_pc(self, 
                    pc_mapping: np.ndarray, pts_xyz: np.ndarray, 
                    reflectance: np.ndarray):
        counts = np.bincount(pc_mapping, minlength = self.target_pillar)
        excess_pillars = np.where(counts > self.target_pc)[0]
        if excess_pillars.size == 0:
            return pc_mapping, pts_xyz, reflectance
        
        keep_mask = np.ones(pc_mapping.shape, dtype = bool)
        for pid in excess_pillars:
            idxs_to_pill        = np.where(pc_mapping == pid)[0]
            drop_idx            = np.random.choice(idxs_to_pill, 
                                                    size = (counts[pid] - self.target_pc),
                                                    replace = False)
            keep_mask[drop_idx] = False

        return(
            pc_mapping[keep_mask],
            pts_xyz[keep_mask],
            reflectance[keep_mask]
        )

    def fast_truncate_pc(self, 
                        pc_mapping: np.ndarray, pts_xyz: np.ndarray, 
                        reflectance: np.ndarray):
        return _truncate_pc_numba(pc_mapping, pts_xyz, reflectance, self.target_pillar, self.target_pc)
        
    def pad_pc(self, 
               pc_mapping: np.ndarray, pts_xyz: np.ndarray, reflectance: np.ndarray, 
               dist_to_centroid: np.ndarray, dist_to_pillar: np.ndarray):
        """Note: padding must be behind lifting calculation to avoid shifting centroid"""

        counts = np.bincount(pc_mapping, minlength = self.target_pillar)
        missing_pillars = np.where(counts < self.target_pc)[0]
        if missing_pillars.size == 0:
            return pc_mapping, pts_xyz, reflectance, dist_to_centroid, dist_to_pillar

        pad_inv           = []
        pad_xyz           = []
        pad_reflectance   = []
        pad_dist_centroid = []
        pad_dist_pillar   = []


        for pid in missing_pillars:
            to_pad = self.target_pc - counts[pid]
            pad_inv           += [np.full(to_pad, pid, dtype = int)]
            pad_xyz           += [np.zeros((to_pad, 3))]
            pad_dist_pillar   += [np.zeros((to_pad, 2))]
            pad_dist_centroid += [np.zeros((to_pad, 3))]
            pad_reflectance   += [np.zeros(to_pad)]

        dummy_inv           = np.concatenate(pad_inv)
        dummy_pts_xyz       = np.vstack(pad_xyz)
        dummy_dist_pillar   = np.vstack(pad_dist_pillar)
        dummy_dist_centroid = np.vstack(pad_dist_centroid)
        dummy_reflectance   = np.concatenate(pad_reflectance)
            
        return (
            np.concatenate([pc_mapping,  dummy_inv]),
            np.vstack([pts_xyz,          dummy_pts_xyz]),
            np.concatenate([reflectance, dummy_reflectance]),
            np.vstack([dist_to_centroid, dummy_dist_centroid]),
            np.vstack([dist_to_pillar,   dummy_dist_pillar])
        )

    
    def fast_pad_pc(self, 
                    pc_mapping: np.ndarray, pts_xyz: np.ndarray, reflectance: np.ndarray, 
                    dist_to_centroid: np.ndarray, dist_to_pillar: np.ndarray):
        """Note: padding must be behind lifting calculation to avoid shifting centroid"""
        counts = np.bincount(pc_mapping, minlength=self.target_pillar)
        missing = np.where(counts < self.target_pc)[0]
        if missing.size == 0:
            return pc_mapping, pts_xyz, reflectance, dist_to_centroid, dist_to_pillar

        missing_counts = self.target_pc - counts[missing]
        total_pad = missing_counts.sum()

        dummy_inv = np.repeat(missing, missing_counts)

        dummy_xyz       = np.zeros((total_pad, 3),  dtype=pts_xyz.dtype)
        dummy_refl      = np.zeros((total_pad,    ), dtype=reflectance.dtype)
        dummy_centroid  = np.zeros((total_pad, 3),  dtype=dist_to_centroid.dtype)
        dummy_pillar    = np.zeros((total_pad, 2),  dtype=dist_to_pillar.dtype)

        return (
            np.concatenate([pc_mapping,    dummy_inv]),
            np.vstack(    [pts_xyz,        dummy_xyz]),
            np.concatenate([reflectance,   dummy_refl]),
            np.vstack(    [dist_to_centroid, dummy_centroid]),
            np.vstack(    [dist_to_pillar,   dummy_pillar])
        )


    def pillar_stats(self, pts_xyz: np.ndarray, pillar_index: np.ndarray, pc_mapping: np.ndarray):
        counts = np.bincount(pc_mapping, minlength = self.target_pillar)
        sum_x  = np.bincount(pc_mapping, weights = pts_xyz[:,0], minlength = self.target_pillar) # Essentially summation per pillar using inverse mapping
        sum_y  = np.bincount(pc_mapping, weights = pts_xyz[:,1], minlength = self.target_pillar) # if the pillar was padded => summation in any axis == 0
        sum_z  = np.bincount(pc_mapping, weights = pts_xyz[:,2], minlength = self.target_pillar)
        centroids = np.vstack((sum_x, sum_y, sum_z)).T / counts[:,None]
        

        centroid_per_point = centroids[pc_mapping]          # shape (n_pts, 3) redistribute the centroid to each point
        dist_to_centroid   = pts_xyz - centroid_per_point

        ix_unique, iy_unique = self.pillar_decoder(pillar_index)
        
        pillar_centers = np.vstack((self.x_centers[ix_unique],
                                    self.y_centers[iy_unique])).T
        pillar_center_per_pt = pillar_centers[pc_mapping]
        dist_to_pillar = pts_xyz[:, :2] - pillar_center_per_pt
        
        return dist_to_centroid, dist_to_pillar








pillar_encoder = lambda x_idx, y_idx, num_y_cell: x_idx * num_y_cell + y_idx
pillar_decoder = lambda pillar_idx, num_y_cell: (pillar_idx // num_y_cell, pillar_idx % num_y_cell)

def truncate_pillar(pillar_index: np.ndarray, pc_mapping: np.ndarray,
                    pts_xyz: np.ndarray, reflectance: np.ndarray,
                    tar_pillar_cnt: int):
    curr_pillar_cnt = pillar_index.shape[0]
    if curr_pillar_cnt <= tar_pillar_cnt:
        return pc_mapping, pts_xyz, reflectance, pillar_index
    
    keep_index = np.random.choice(curr_pillar_cnt, size = tar_pillar_cnt, replace = False)
    
    remap = -np.ones(curr_pillar_cnt, dtype=int)
    remap[keep_index] = np.arange(tar_pillar_cnt)
    new_inv = remap[pc_mapping]

    keep_pts_mask = new_inv >= 0
    
    return (
        new_inv[keep_pts_mask],
        pts_xyz[keep_pts_mask],        
        reflectance[keep_pts_mask],
        pillar_index[keep_index]
    )

def pad_pillars(pillar_index: np.ndarray,
                pc_mapping:   np.ndarray,
                pts_xyz:      np.ndarray,
                reflectance:  np.ndarray,
                num_x:        int,
                num_y:        int,
                tar_pillar_cnt:int,
                tar_pc_cnt:    int):
    curr_cnt = pillar_index.shape[0]
    if curr_cnt >= tar_pillar_cnt:
        return pc_mapping, pts_xyz, reflectance, pillar_index

    occ = np.zeros((num_x, num_y), dtype=bool)
    ix_u, iy_u = pillar_decoder(pillar_index, num_y)
    occ[ix_u, iy_u] = True
    empty_coords = np.argwhere(~occ)  

    pad_cnt = tar_pillar_cnt - curr_cnt
    sel = empty_coords[np.random.choice(
        empty_coords.shape[0],
        size=pad_cnt,
        replace=False
    )]
    pad_globals = pillar_encoder(sel[:,0], sel[:,1], num_y)

    new_slots  = np.arange(curr_cnt, curr_cnt + pad_cnt)
    dummy_inv  = np.repeat(new_slots, tar_pc_cnt)
    dummy_xyz  = np.zeros((pad_cnt * tar_pc_cnt, 3))
    dummy_ref  = np.zeros((pad_cnt * tar_pc_cnt,))

    return (
        np.concatenate([pc_mapping,   dummy_inv]),
        np.vstack([pts_xyz,           dummy_xyz]),
        np.concatenate([reflectance,  dummy_ref]),
        np.concatenate([pillar_index, pad_globals])                        
    )
    
def truncate_pc(pc_mapping: np.ndarray, pts_xyz: np.ndarray, 
                reflectance: np.ndarray, tar_pillar_cnt: int, 
                tar_pc_cnt: int):
    counts = np.bincount(pc_mapping, minlength = tar_pillar_cnt)
    excess_pillars = np.where(counts > tar_pc_cnt)[0]
    if excess_pillars.size == 0:
        return pc_mapping, pts_xyz, reflectance
    
    keep_mask = np.ones(pc_mapping.shape, dtype = bool)
    for pid in excess_pillars:
        idxs_to_pill        = np.where(pc_mapping == pid)[0]
        drop_idx            = np.random.choice(idxs_to_pill, 
                                                size = (counts[pid] - tar_pc_cnt),
                                                replace = False)
        keep_mask[drop_idx] = False

    return(
        pc_mapping[keep_mask],
        pts_xyz[keep_mask],
        reflectance[keep_mask]
    )
    
def pad_pc(pc_mapping: np.ndarray, pts_xyz: np.ndarray, reflectance: np.ndarray, 
           dist_to_centroid: np.ndarray, dist_to_pillar: np.ndarray, 
           tar_pillar_cnt: int, tar_pc_cnt: int):
    """Note: padding must be behind lifting calculation to avoid shifting centroid"""

    counts = np.bincount(pc_mapping, minlength = tar_pillar_cnt)
    missing_pillars = np.where(counts < tar_pc_cnt)[0]
    if missing_pillars.size == 0:
        return pc_mapping, pts_xyz, reflectance, dist_to_centroid, dist_to_pillar

    pad_inv           = []
    pad_xyz           = []
    pad_reflectance   = []
    pad_dist_centroid = []
    pad_dist_pillar   = []


    for pid in missing_pillars:
        to_pad = tar_pc_cnt - counts[pid]
        pad_inv           += [np.full(to_pad, pid, dtype = int)]
        pad_xyz           += [np.zeros((to_pad, 3))]
        pad_dist_pillar   += [np.zeros((to_pad, 2))]
        pad_dist_centroid += [np.zeros((to_pad, 3))]
        pad_reflectance   += [np.zeros(to_pad)]

    dummy_inv           = np.concatenate(pad_inv)
    dummy_pts_xyz       = np.vstack(pad_xyz)
    dummy_dist_pillar   = np.vstack(pad_dist_pillar)
    dummy_dist_centroid = np.vstack(pad_dist_centroid)
    dummy_reflectance   = np.concatenate(pad_reflectance)
        
    return (
        np.concatenate([pc_mapping,  dummy_inv]),
        np.vstack([pts_xyz,          dummy_pts_xyz]),
        np.concatenate([reflectance, dummy_reflectance]),
        np.vstack([dist_to_centroid, dummy_dist_centroid]),
        np.vstack([dist_to_pillar,   dummy_dist_pillar])
    )


def pillar_stats(pts_xyz: np.ndarray, pillar_index: np.ndarray, pc_mapping: np.ndarray, num_y: int, x_centers: np.ndarray, y_centers: np.ndarray, tar_pillar_cnt: int):
    counts = np.bincount(pc_mapping, minlength = tar_pillar_cnt)
    sum_x  = np.bincount(pc_mapping, weights = pts_xyz[:,0], minlength = tar_pillar_cnt) # Essentially summation per pillar using inverse mapping
    sum_y  = np.bincount(pc_mapping, weights = pts_xyz[:,1], minlength = tar_pillar_cnt) # if the pillar was padded => summation in any axis == 0
    sum_z  = np.bincount(pc_mapping, weights = pts_xyz[:,2], minlength = tar_pillar_cnt)
    centroids = np.vstack((sum_x, sum_y, sum_z)).T / counts[:,None]
    

    centroid_per_point = centroids[pc_mapping]          # shape (n_pts, 3) redistribute the centroid to each point
    dist_to_centroid   = pts_xyz - centroid_per_point

    ix_unique, iy_unique = pillar_decoder(pillar_index, num_y)
    
    pillar_centers = np.vstack((x_centers[ix_unique],
                                y_centers[iy_unique])).T
    pillar_center_per_pt = pillar_centers[pc_mapping]
    dist_to_pillar = pts_xyz[:, :2] - pillar_center_per_pt
    
    return dist_to_centroid, dist_to_pillar