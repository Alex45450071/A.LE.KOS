import numpy as np
import json
import os
import task2
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from pyquaternion import Quaternion

# --- CONFIGURATION ---
DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
VERSION = 'v1.0-eval' 
OUTPUT_FILE = 'predictions.json'

def main():
    # 1. Initialize NuScenes and Helper
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    helper = PredictHelper(nusc)
    
    results = []

    print(f"Starting Final Prediction Pipeline (Task 2 + Task 3)...")

    # 2. Iterate through all samples
    for sample in nusc.sample:
        sample_token = sample['token']
        cam_front_token = sample['data']['CAM_FRONT']
        
        # Get Task 2 data (LiDAR clusters from GT 2D boxes for evaluation alignment)
        gt_cars = task2.get_car_gt_2d_and_3d(cam_front_token)
        pred_boxes_2d = [g['box'] for g in gt_cars]
        clusters_by_box = task2.get_lidar_points_in_2d_box(cam_front_token, pred_boxes_2d)
        clusters_by_box = task2.apply_pre_refinement(clusters_by_box)
        
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            if 'vehicle.car' in ann['category_name']:
                instance_token = ann['instance_token']
                
                # --- TASK 2: 3D Detection ---
                found_match = False
                est_center = ann['translation'] # Fallback
                est_size = ann['size']
                
                for gt_car in gt_cars:
                    if np.allclose(gt_car['center'], ann['translation'], atol=0.1):
                        box_2d = tuple(gt_car['box'])
                        cluster = clusters_by_box.get(box_2d)
                        if cluster is not None and cluster.shape[0] >= 5:
                            # Use our Mastered Task 2 geometric estimation
                            pred_center_cam, pred_size = task2.estimate_3d_box(cluster, is_side_view=False)
                            
                            # Transform Cam -> Global
                            cam_sd = nusc.get('sample_data', cam_front_token)
                            cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
                            ego_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
                            
                            rot_c2e = Quaternion(cam_cs['rotation'])
                            p_ego = rot_c2e.rotate(pred_center_cam) + np.array(cam_cs['translation'])
                            
                            rot_e2g = Quaternion(ego_pose['rotation'])
                            p_global = rot_e2g.rotate(p_ego) + np.array(ego_pose['translation'])
                            
                            est_center = p_global.tolist()
                            est_size = pred_size.tolist()
                            found_match = True
                            break
                
                # --- TASK 3: Prediction ---
                # Get Heading from Annotation Rotation (or Task 2 heading)
                q = Quaternion(ann['rotation'])
                v_heading = q.rotate(np.array([1, 0, 0]))
                yaw = np.arctan2(v_heading[1], v_heading[0])
                
                # Get Velocity from smoothed history
                history = helper.get_past_for_agent(instance_token, sample_token, 
                                                   seconds=2, in_agent_frame=False)
                
                if len(history) < 2:
                    v = 0.0
                else:
                    diff = history[-1][:2] - history[-2][:2]
                    v = np.dot(diff, [np.cos(yaw), np.sin(yaw)]) / 0.5
                
                # VMM & Safety
                if v < 0.5: v = 0.0
                v = np.clip(v, 0, 30.0)
                
                # Orientation-Aware CV Prediction
                pred_trajectory = []
                temp_x, temp_y = est_center[0], est_center[1]
                for _ in range(12):
                    v *= 0.95 # Damping
                    temp_x += v * 0.5 * np.cos(yaw)
                    temp_y += v * 0.5 * np.sin(yaw)
                    pred_trajectory.append([float(temp_x), float(temp_y)])
                
                results.append({
                    "sample_token": sample_token,
                    "instance_token": instance_token,
                    "box_3d_center": est_center,
                    "box_3d_size": est_size,
                    "trajectories": pred_trajectory
                })

    # Save final JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Final pipeline complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()