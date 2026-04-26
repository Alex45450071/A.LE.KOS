from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
nusc = NuScenes(version="v1.0-eval", dataroot=DATAROOT, verbose=False)

sample = nusc.sample[0]

try:
    # Aggregating LIDAR_TOP into CAM_FRONT coordinate frame
    pc, times = LidarPointCloud.from_file_multisweep(nusc, sample, "LIDAR_TOP", "CAM_FRONT", nsweeps=10)
    print(f"Successfully loaded multisweep. Points: {pc.points.shape[1]}")
except Exception as e:
    print(f"Error loading multisweep: {e}")
