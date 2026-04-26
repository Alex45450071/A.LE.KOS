import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

# --- CONFIGURATION ---
DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
VERSION = 'v1.0-eval'
PREDICTIONS_FILE = 'predictions.json'

def evaluate_predictions():
    # 1. Load NuScenes and Helper
    print("Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    helper = PredictHelper(nusc)
    
    # 2. Load Predictions
    with open(PREDICTIONS_FILE, 'r') as f:
        predictions = json.load(f)
    
    all_ade = []
    all_fde = []
    skipped_no_future = 0
    
    print(f"Evaluating {len(predictions)} predictions...")
    
    # To speed up, we only evaluate a subset if it's too many
    # But for 5k it should be okay in a few seconds.
    for i, pred in enumerate(predictions):
        sample_token = pred['sample_token']
        instance_token = pred['instance_token']
        pred_traj = np.array(pred['trajectories']) # (12, 2)
        
        # 3. Get Ground Truth Future (6 seconds = 12 frames)
        # in_agent_frame=False gives Global coords to match our predictions.json
        gt_future = helper.get_future_for_agent(instance_token, sample_token, 
                                               seconds=6, in_agent_frame=False)
        
        if len(gt_future) == 0:
            skipped_no_future += 1
            continue
            
        # 4. Align steps
        # If GT is shorter (vehicle left scene), we only compare available steps
        num_steps = min(len(pred_traj), len(gt_future))
        pred_segment = pred_traj[:num_steps]
        gt_segment = gt_future[:num_steps, :2] # Only X, Y
        
        # 5. Calculate ADE (Average Displacement Error)
        distances = np.linalg.norm(pred_segment - gt_segment, axis=1)
        ade = np.mean(distances)
        
        # 6. Calculate FDE (Final Displacement Error)
        fde = distances[-1]
        
        all_ade.append(ade)
        all_fde.append(fde)
        
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1}/{len(predictions)}...")

    # 7. Summary
    if len(all_ade) == 0:
        print("No matches found for evaluation!")
        return

    mean_ade = np.mean(all_ade)
    mean_fde = np.mean(all_fde)
    median_ade = np.median(all_ade)
    
    print("\n--- TASK 3 EVALUATION (ADE/FDE) ---")
    print(f"Total Evaluated: {len(all_ade)}")
    print(f"Skipped (No Future): {skipped_no_future}")
    print(f"Mean ADE: {mean_ade:.3f} m")
    print(f"Mean FDE: {mean_fde:.3f} m")
    print(f"Median ADE: {median_ade:.3f} m")
    
    return {
        "mean_ade": mean_ade,
        "mean_fde": mean_fde,
        "median_ade": median_ade
    }

if __name__ == "__main__":
    evaluate_predictions()
