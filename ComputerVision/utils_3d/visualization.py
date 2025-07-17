import time
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

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