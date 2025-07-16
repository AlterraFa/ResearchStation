import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import struct
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_kitti_bin(pc_path: str) -> np.ndarray:
    """
    Load a KITTI Velodyne .bin file into an (N,4) numpy array
    where each point is [x, y, z, reflectance].
    """
    # read raw bytes
    points = np.fromfile(pc_path, dtype=np.float32)
    # reshape into N×4
    return points.reshape(-1, 4)

def visualize_pointcloud(pc: np.ndarray):
    """
    Given an (N,4) array, create an Open3D point cloud and visualize it.
    """
    # drop reflectance
    xyz = pc[:, :3]
    # make Open3D PointCloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    # optional: color by height (z)
    z = xyz[:, 2]
    z_norm = (z - z.min()) / (np.ptp(z) + 1e-6)
    colors = plt.get_cmap('viridis')(z_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # visualize
    o3d.visualization.draw_geometries([pcd],
        window_name="KITTI Lidar",
        width=800, height=600,
        left=50, top=50,
        point_show_normal=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("bin_file", help="Path to KITTI .bin pointcloud")
    args = parser.parse_args()

    pc = load_kitti_bin(args.bin_file)
    visualize_pointcloud(pc)
