import os, sys
import glob
import time
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

def load_bin(pcPath: str) -> np.ndarray:
    data = np.fromfile(pcPath, dtype = np.float32)
    return data.reshape(-1, 4)

def colorized_by_z(xyz: np.ndarray) -> np.ndarray: 
    z = xyz[:, 2]
    zNorm = (z - z.min()) / (np.ptp(z) + 1e-6)
    return plt.get_cmap('viridis')(zNorm)[:, :3]

def update(vis: o3d.visualization.Visualizer, 
           pcd: o3d.geometry.PointCloud,
           delay: int):
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(delay)
    
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
    return corners3d

def create_line_set(corners: np.ndarray, color=(1,0,0)) -> o3d.geometry.LineSet:
    """Build an Open3D LineSet from 8 corners (any coord frame)."""
    # 12 edges of a box
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    colors = [color for _ in edges]
    ls = o3d.geometry.LineSet(
        points  = o3d.utility.Vector3dVector(corners),
        lines   = o3d.utility.Vector2iVector(edges)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls



def visualize_seq(train_folder: str, fps: float = 10) -> None:
    frame_delay = 1 / fps

    frame_files = sorted(glob.glob(train_folder + "/velodyne/*"))
    calib_files = sorted(glob.glob(train_folder + "/calib/*"))
    label_files = sorted(glob.glob(train_folder + "/label_2/*"))
    if not frame_files or len(frame_files) != len(calib_files) or len(frame_files) != len(label_files): 
        print("Invalid folder")
        exit(-1)
    
    firstPc = load_bin(frame_files[0])
    xyz = firstPc[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colorized_by_z(xyz))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name = "KITTI data loader",
        width = 800, height = 600,
        left = 50, top = 50
    )
    update(vis, pcd, frame_delay)
    
    box_geometry: list[o3d.geometry.LineSet] = []

    bar = tqdm(frame_files, desc="Playing frames", unit="frame")
    for _ in bar:
        # Load calibration, binary and labels
        pc     = load_bin(frame_files[bar.n])
        T      = load_calib(calib_files[bar.n])
        labels = load_labels(label_files[bar.n])
        
        xyz = pc[:, :3]
        # Remove old bbox
        for bg in box_geometry:
            vis.remove_geometry(bg, reset_bounding_box = False)
        box_geometry.clear()

        cam_to_Vel = np.linalg.inv(T)
        
        for objs in labels:
            corner_cam = get_3D_box_corneres(objs) # Camera corners
            corner_Vel = (cam_to_Vel @ corner_cam.T).T[:, :3]
            line_set   = create_line_set(corner_Vel)
            vis.add_geometry(line_set, reset_bounding_box = False)
            box_geometry.append(line_set)

        pcd.points = o3d.utility.Vector3dVector(xyz)
        colors     = colorized_by_z(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        update(vis, pcd, frame_delay)
    vis.destroy_window()
    
def inspect_frame(index: int, train_folder: str) -> None:
    """Load one .bin + its calib & label, show pointcloud with boxes."""
    
    frame_files = sorted(glob.glob(train_folder + "/velodyne/*"))

    bin_file  = frame_files[index]
    idx       = os.path.splitext(os.path.basename(bin_file))[0]
    calibFile = os.path.join(train_folder, 'calib',    f"{idx}.txt")
    labelFile = os.path.join(train_folder, 'label_2',  f"{idx}.txt")

    pc    = load_bin(bin_file)[:, :3]
    Tcam  = load_calib(calibFile)
    labels= load_labels(labelFile)
    camToVel = np.linalg.inv(Tcam)

    # build pcd and boxes
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pc),
    )
    pcd.colors = o3d.utility.Vector3dVector(colorized_by_z(pc))
    boxGeoms = []
    for obj in labels:
        corners_cam = get_3D_box_corneres(obj)
        corners_Vel = (camToVel @ corners_cam.T).T[:, :3]
        boxGeoms.append(create_line_set(corners_Vel))

    o3d.visualization.draw_geometries(
        [pcd, *boxGeoms],
        window_name=f"Inspect frame {idx}"
    )




if __name__ == "__main__":
    binFolder = "./dataset/training"
    
    p = argparse.ArgumentParser(
        description="KITTI viewer: full sequence or single-frame inspect"
    )
    p.add_argument("train_folder",
        help="root containing velodyne/, calib/, label_2/")
    p.add_argument("--fps", type=float, default=None,
        help="playback speed for sequence")
    p.add_argument("--inspect", type=int, default=None,
        help="Index of scene to inspect")
    args = p.parse_args()
    
    if args.fps and args.inspect:
        print("Inspection mode and visualization mode were specified. Exiting ...")
        exit(2134)
    elif args.inspect: 
        inspect_frame(args.inspect, args.train_folder)
    elif args.fps:
        visualize_seq(args.train_folder, args.fps)