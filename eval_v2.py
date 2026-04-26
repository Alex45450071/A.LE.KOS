import task2
import os
import json
import numpy as np

def run_eval(num_samples=20):
    all_cam_tokens = []
    for s in task2.nusc.scene:
        st = s["first_sample_token"]
        while st:
            samp = task2.nusc.get("sample", st)
            all_cam_tokens.append(samp["data"]["CAM_FRONT"])
            st = samp["next"]
            
    import random
    random.seed(42)  
    random.shuffle(all_cam_tokens)
    eval_tokens = all_cam_tokens[:num_samples]
    
    print(f"Starting evaluation on {num_samples} samples...")
    results = task2.evaluate_task2_mean_center_error(eval_tokens, iou_threshold=0.5)
    
    print("\n--- Evaluation Summary ---")
    print(f"Matched Cars: {results['matched_cars_total']}")
    print(f"MCE Median: {results['mean_center_error_median_m']:.3f} m")
    print(f"MCE Robust: {results['mean_center_error_robust_m']:.3f} m")
    print(f"MCE Depth Trim: {results['mean_center_error_depth_trim_m']:.3f} m")
    
    return results

if __name__ == "__main__":
    run_eval(20)
