import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from ultralytics import YOLO

# 1. Initialize NuScenes (point this to your extracted folder)
DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
nusc = NuScenes(version='v1.0-eval', dataroot=DATAROOT, verbose=False)

# 2. Load YOLOv8 model
yolo_model = YOLO("yolov8m.pt")

# Allowed YOLO classes for this challenge:
# car=2, motorcycle=3, bus=5, truck=7
ALLOWED_YOLO_CLASS_IDS = {2, 3, 5, 7}


def get_2d_ground_truth(nusc, sample_data_token):
    """
    Project visible 3D NuScenes boxes to 2D camera-space boxes.
    Returns:
        [
            {'box': [xmin, ymin, xmax, ymax], 'class': '<nuScenes class name>'},
            ...
        ]
    """
    sample_data = nusc.get("sample_data", sample_data_token)
    cam_intrinsic = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])["camera_intrinsic"]
    cam_intrinsic = np.array(cam_intrinsic)

    image_path, boxes_3d, _ = nusc.get_sample_data(sample_data_token)
    image = Image.open(image_path)
    image_w, image_h = image.size

    gt_boxes = []
    for box in boxes_3d:
        # Keep existing vehicle-only filtering behavior.
        if "vehicle" not in box.name:
            continue
        # Only score vehicles that are at least ~40% visible.
        ann_token = getattr(box, "token", None)
        if ann_token is not None:
            ann = nusc.get("sample_annotation", ann_token)
            visibility_token = int(ann.get("visibility_token", "0"))
            if visibility_token <= 1:
                continue

        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
        xmin, ymin = np.min(corners_2d, axis=1)
        xmax, ymax = np.max(corners_2d, axis=1)

        xmin = max(0.0, float(xmin))
        ymin = max(0.0, float(ymin))
        xmax = min(float(image_w), float(xmax))
        ymax = min(float(image_h), float(ymax))

        if xmax <= xmin or ymax <= ymin:
            continue

        gt_boxes.append({"box": [xmin, ymin, xmax, ymax], "class": box.name})

    return gt_boxes


def get_yolo_predictions(image_path):
    """
    Run YOLOv8 inference and return filtered detections:
        [{'box': [xmin, ymin, xmax, ymax], 'class': <class_id>, 'confidence': <float>}, ...]
    """
    results = yolo_model.predict(
        image_path,
        imgsz=1280,
        conf=0.15,
        iou=0.5,
        verbose=False,
    )
    detections = []

    for result in results:
        boxes = result.boxes
        for det in boxes:
            class_id = int(det.cls.item())
            if class_id not in ALLOWED_YOLO_CLASS_IDS:
                continue

            xyxy = det.xyxy[0].tolist()
            confidence = float(det.conf.item())
            detections.append(
                {
                    "box": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                    "class": class_id,
                    "confidence": confidence,
                }
            )

    return detections


def calculate_iou(boxA, boxB):
    """Calculate IoU between two [xmin, ymin, xmax, ymax] boxes."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def score_frame(gt_boxes, pred_boxes, match_threshold=0.4):
    """
    For each GT box, find the prediction with the highest IoU.
    A GT box is a missed detection if best IoU <= match_threshold.
    Returns mean IoU over GT boxes and missed detection count.
    """
    if not gt_boxes:
        return 0.0, 0

    best_ious = []
    missed_detections = 0

    for gt in gt_boxes:
        gt_box = gt["box"]
        best_iou = 0.0

        for pred in pred_boxes:
            iou = calculate_iou(gt_box, pred["box"])
            if iou > best_iou:
                best_iou = iou

        best_ious.append(best_iou)
        if best_iou <= match_threshold:
            missed_detections += 1

    mean_iou = float(np.mean(best_ious))
    return mean_iou, missed_detections


if __name__ == "__main__":
    # 3. Loop through first 5 samples of the first scene
    scene = nusc.scene[0]
    sample_token = scene["first_sample_token"]

    for idx in range(5):
        if not sample_token:
            break

        sample = nusc.get("sample", sample_token)
        cam_front_token = sample["data"]["CAM_FRONT"]
        cam_front_data = nusc.get("sample_data", cam_front_token)
        image_path = nusc.get_sample_data_path(cam_front_token)

        gt_boxes = get_2d_ground_truth(nusc, cam_front_token)
        yolo_boxes = get_yolo_predictions(image_path)
        mean_iou, missed = score_frame(gt_boxes, yolo_boxes, match_threshold=0.4)

        print(
            f"Sample {idx + 1} ({cam_front_data['filename']}): "
            f"GT boxes = {len(gt_boxes)} | YOLO detections = {len(yolo_boxes)} | "
            f"Mean IoU = {mean_iou:.3f} | Missed Detections = {missed}"
        )

        # 4. Visualization: GT in green, YOLO predictions in red
        image = Image.open(image_path)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(image)

        for gt in gt_boxes:
            xmin, ymin, xmax, ymax = gt["box"]
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                max(0.0, ymin - 5),
                "GT",
                color="green",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )

        for pred in yolo_boxes:
            xmin, ymin, xmax, ymax = pred["box"]
            conf = pred.get("confidence", 0.0)
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                max(0.0, ymin - 5),
                f"Pred {conf:.2f}",
                color="red",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )

        legend_handles = [
            Patch(edgecolor="green", facecolor="none", linewidth=2, label="Ground Truth"),
            Patch(edgecolor="red", facecolor="none", linewidth=2, label="YOLO Prediction"),
        ]
        ax.legend(handles=legend_handles, loc="upper right")
        ax.set_title(f"Sample {idx + 1}: CAM_FRONT")
        ax.axis("off")

        output_path = f"output_sample_{idx + 1}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        sample_token = sample["next"]