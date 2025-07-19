import numpy as np
import glob

from tqdm.auto import tqdm

def load_bin(pcPath: str) -> np.ndarray:
    data = np.fromfile(pcPath, dtype = np.float32)
    return data.reshape(-1, 4)

    
def load_labels(label_path: str) -> list[dict]:
    """Parse KITTI label file into a list of dicts with dims, loc, rot_y."""
    objs = []
    with open(label_path, 'r') as f:
        for line in f:
            data = line.split()
            cls = data[0]
            if cls == 'DontCare': 
                continue
            h, w, l = map(float, data[8:11])
            x, y, z = map(float, data[11:14])
            rot_y = float(data[14])
            objs.append({'h':h, 'w':w, 'l':l,
                         'x':x, 'y':y, 'z':z,
                         'rot_y':rot_y})
    return objs

def load_calib(calibPath: str) -> np.ndarray:
    with open(calibPath, 'r') as f:
        for line in f:
            if line.startswith('Tr_velo_to_cam:'):
                num_str = line.split(':', 1)[1].strip()
                vals = np.fromstring(num_str, sep=' ')
                T = np.eye(4, dtype=np.float32)
                T[:3, :4] = vals.reshape(3, 4)
                return T
    raise FileNotFoundError(f"No Tr_velo_to_cam in {calibPath}")

def save_bin(pc: np.ndarray, filePath: str) -> None:
    assert pc.ndim == 2 and pc.shape[1] == 4, "pc must be N×4"
    pc.astype(np.float32).tofile(filePath)

def load_multi_pc(load_folder: str, from_idx: int = 0, to_idx: int = -1) -> list[np.ndarray]:
    file_paths = glob.glob(load_folder + "/*")
    
    assert to_idx < len(file_paths), f"List index out of range"
    
    pc_part = []
    for path in tqdm(file_paths[from_idx: to_idx], desc = "Loading point clouds", unit = " Files"):
        pc_part += [load_bin(path)]
        
    return pc_part