import numpy as np

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