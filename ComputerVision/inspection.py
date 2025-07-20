import os, sys
import glob
import argparse
import configparser
import numpy as np
import open3d as o3d

from tqdm.auto import tqdm
from utils_3d.disk import *
from utils_3d.visualization import *
from utils_3d.math_helper import *


config = configparser.ConfigParser()
config.read("./model/config.ini")

dimension_arg = config["Dimension"]
pillar_arg    = config["Pillarization"]

xmin       = dimension_arg.getfloat("xmin")
xmax       = dimension_arg.getfloat("xmax")
ymin       = dimension_arg.getfloat("ymin")
ymax       = dimension_arg.getfloat("ymax")
zmax       = dimension_arg.getfloat("zmax")
zmin       = dimension_arg.getfloat("zmin")
resolution = dimension_arg.getfloat("resolution")

# Running main
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
    
def inspect_frame(index: int, train_folder: str, crop_mode: bool) -> None:
    """Load one .bin + its calib & label, show pointcloud with boxes."""
    
    frame_files = sorted(glob.glob(train_folder + "/velodyne/*"))

    bin_file  = frame_files[index]
    idx       = os.path.splitext(os.path.basename(bin_file))[0]
    calibFile = os.path.join(train_folder, 'calib',    f"{idx}.txt")
    labelFile = os.path.join(train_folder, 'label_2',  f"{idx}.txt")

    pc       = load_bin(bin_file)[:, :3]
    Tcam     = load_calib(calibFile)
    labels   = load_labels(labelFile)
    camToVel = np.linalg.inv(Tcam)

    if crop_mode:
        pc = crop_pc(pc, xmax = xmax, xmin = xmin, ymax = ymax, ymin = ymin, zmax = zmax, zmin = zmin)

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

    # Origin point
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius = 0.2)
    origin_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    origin_sphere.translate([0, 0, 0])

    o3d.visualization.draw_geometries(
        [pcd, *boxGeoms, origin_sphere],
        window_name=f"Inspect frame {idx}"
    )




if __name__ == "__main__":
    
    p = argparse.ArgumentParser(
        description="KITTI viewer: full sequence or single-frame inspect"
    )
    p.add_argument("train_folder",
        help = "root containing velodyne/, calib/, label_2/")
    p.add_argument("--fps", type=float, default=None,
        help = "playback speed for sequence")
    p.add_argument("--inspect", type=int, default=None,
        help = "index of scene to inspect")
    p.add_argument("--enable_crop", action = "store_true", dest="crop_mode",
        help = "only crop the forward region (default: False)")
    p.add_argument("--disable_crop", action = "store_false", dest="crop_mode",
        help = "disable cropping")
    p.set_defaults(crop_mode = False)
    args = p.parse_args()
    
    if args.fps and args.inspect:
        print("Inspection mode and visualization mode were specified. Exiting ...")
        exit(2134)
    elif args.inspect: 
        inspect_frame(args.inspect, args.train_folder, args.crop_mode)
    elif args.fps:
        visualize_seq(args.train_folder, args.fps)