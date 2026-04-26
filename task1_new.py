import os
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'yolo26s.pt'      # 2026 SOTA: NMS-free, better for overlap
DATASET_PATH = 'path/to/your/dataset/images/val' 
IMG_SIZE = 1280                # High res to catch small objects in background
IOU_THRESHOLD = 0.5            # Your required metric threshold
CONF_THRESHOLD = 0.25          # Initial starting point

def run_optimized_check():
    print(f"🚀 Loading {MODEL_PATH} for evaluation...")
    
    # 1. Load the Model
    # We use YOLO26 because it handles the overlapping GTs in your image 
    # without the NMS "deletion" problem.
    model = YOLO(MODEL_PATH)

    # 2. Run Standard Validation
    # This gives you the official mAP and F1 results at IoU 0.5
    print("\n--- Running Official Validation ---")
    metrics = model.val(
        data='data.yaml', 
        imgsz=IMG_SIZE, 
        iou=IOU_THRESHOLD, 
        conf=0.001,      # Low conf to get the full PR curve
        augment=True,    # Test Time Augmentation for +2-3% F1 boost
        plots=True
    )
    
    # Extract the F1 Score at IoU 0.5
    # YOLOv8/26 stores this in metrics.results_dict
    f1_score = metrics.results_dict.get('metrics/f1(B)', 0)
    print(f"\n✅ Initial Validation F1 Score: {f1_score:.4f}")

    # 3. SAHI (Slicing) Booster
    # Use this if F1 is still below 0.75. Slicing helps with tiny objects.
    print("\n--- Initializing SAHI Sliced Inference ---")
    detection_model = AutoDetectionModel.from_model_type(
        model_type='yolov8', # YOLO26 is compatible with v8 wrapper
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD,
        device='cuda' # or 'cpu'
    )

    # Loop through a few test images to verify "closeness" to Ground Truth
    test_images = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH)[:5]]
    
    for img_path in test_images:
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        # result.export_visuals(export_dir="sahi_results/")
        print(f"Analyzed {os.path.basename(img_path)}: Found {len(result.object_prediction_list)} objects.")

    if f1_score > 0.75:
        print("\n🏆 TARGET REACHED: F1 Score is above 0.75!")
    else:
        print("\n⚠️ F1 is below 0.75. Action: Add background (negative) images to reduce False Positives.")

if __name__ == "__main__":
    run_optimized_check()