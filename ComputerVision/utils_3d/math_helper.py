import numpy as np

def get_3D_box_corneres(obj: dict) -> np.ndarray:
    """
    Returns an (8,3) array of corner points in camera coords.
    Follows KITTI convention: y is down, box bottom at y.
    """
    l, w, h = obj['l'], obj['w'], obj['h']
    x, y, z = obj['x'], obj['y'], obj['z']

    # corners in the object frame (at (0, 0, 0))
    x_c = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_c = [   0 ,    0 ,    0 ,    0 ,   -h ,   -h ,   -h ,   -h ]
    z_c = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    corners = np.vstack((x_c, y_c, z_c, np.ones_like(x_c)))
    
    # rotation around Y (meaning rotating around v)
    ry = obj['rot_y']
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry), x],
        [          0, 1,          0, y],
        [-np.sin(ry), 0, np.cos(ry), z],
        [0          , 0, 0         , 1]
    ], dtype=np.float32)
    corners3d = (R @ corners).T
    return corners3d, np.array([l, w, h])

def compute_heading(corner1: np.ndarray, corner2: np.ndarray):
    dx = corner1[0] - corner2[0]
    dy = corner1[1] - corner2[1]
    
    raw_yaw  = np.arctan2(dy, dx)
    return (raw_yaw + np.pi) % np.pi

def crop_pc(pc: np.ndarray, 
            xmin: float = 0  , xmax: float = 70,
            ymin: float = -40, ymax: float = 40,
            zmin: float = -3 , zmax: float = 1) -> np.ndarray:
    mask = (
        (pc[:, 0] > xmin) & (pc[:, 0] < xmax) &
        (pc[:, 1] > ymin) & (pc[:, 1] < ymax) &
        (pc[:, 2] > zmin) & (pc[:, 2] < zmax)
    )
    return pc[mask]