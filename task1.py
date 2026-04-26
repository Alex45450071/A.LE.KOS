import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

# 1. Initialize NuScenes (point this to your extracted folder)
DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
nusc = NuScenes(version='v1.0-eval', dataroot=DATAROOT, verbose=False)

# 2. Load YOLOv8m model through SAHI wrapper
sahi_detection_model = UltralyticsDetectionModel(
    model_path="yolov8m.pt",
    confidence_threshold=0.36,
)

# COCO IDs for strict vehicle class handling.
YOLO_ID_TO_CLASS = {
    2: "car",
    5: "bus",
    7: "truck",
}
ALLOWED_VEHICLE_CLASSES = {"car", "truck", "bus"}
# Precision-first configuration (minimize false positives).
GLOBAL_MIN_CONFIDENCE = 0.42
FAR_OBJECT_MAX_HEIGHT_PX = 55.0
FAR_OBJECT_MIN_CONFIDENCE = 0.56
MIN_BOX_WIDTH_PX = 8.0
MIN_BOX_HEIGHT_PX = 8.0
ROAD_ROI_START_RATIO = 0.49
MERGE_IOU_THRESHOLD = 0.36
MERGE_CONTAINMENT_THRESHOLD = 0.62


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
    Hybrid SAHI + standard YOLOv8 inference.
    Returns detections in format:
        [xmin, ymin, xmax, ymax, confidence, class_name]
    """
    image = Image.open(image_path)
    image_w, image_h = image.size
    road_ymin = int(image_h * ROAD_ROI_START_RATIO)

    # Pass 1: strict global hybrid prediction for stable near-object detections.
    global_result = get_sliced_prediction(
        image=image_path,
        detection_model=sahi_detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=True,
        postprocess_type="GREEDYNMM",
        postprocess_match_threshold=0.45,
        verbose=0,
    )

    # Pass 2: dynamic road-ROI pass with finer slicing to recover far objects.
    roi_array = np.array(image)[road_ymin:, :, :]
    roi_result = get_sliced_prediction(
        image=roi_array,
        detection_model=sahi_detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        perform_standard_pred=False,
        postprocess_type="GREEDYNMM",
        postprocess_match_threshold=0.45,
        verbose=0,
    )

    raw_detections = []

    def collect_predictions(predictions, y_offset=0.0):
        for pred in predictions:
            class_id = int(pred.category.id)
            class_name = pred.category.name.lower()

            if class_id in YOLO_ID_TO_CLASS:
                class_name = YOLO_ID_TO_CLASS[class_id]
            elif class_name == "van":
                class_name = "car"

            if class_name not in ALLOWED_VEHICLE_CLASSES:
                continue

            bbox = pred.bbox
            xmin = float(bbox.minx)
            ymin = float(bbox.miny) + y_offset
            xmax = float(bbox.maxx)
            ymax = float(bbox.maxy) + y_offset
            confidence = float(pred.score.value)
            if confidence < GLOBAL_MIN_CONFIDENCE:
                continue

            box_height = ymax - ymin
            box_width = xmax - xmin

            # Remove tiny boxes that are usually noisy false positives.
            if box_width < MIN_BOX_WIDTH_PX or box_height < MIN_BOX_HEIGHT_PX:
                continue

            # Far-away objects are usually tiny and noisier; require higher confidence.
            if box_height <= FAR_OBJECT_MAX_HEIGHT_PX and confidence < FAR_OBJECT_MIN_CONFIDENCE:
                continue

            raw_detections.append([xmin, ymin, xmax, ymax, confidence, class_name])

    collect_predictions(global_result.object_prediction_list, y_offset=0.0)
    collect_predictions(roi_result.object_prediction_list, y_offset=float(road_ymin))

    # Class-wise dedup to suppress global+ROI double detections.
    detections = []
    for cls_name in ALLOWED_VEHICLE_CLASSES:
        cls_dets = [d for d in raw_detections if d[5] == cls_name]
        cls_dets.sort(key=lambda d: d[4], reverse=True)
        kept = []
        for det in cls_dets:
            is_duplicate = False
            for k in kept:
                iou = calculate_iou(det[:4], k[:4])
                containment = calculate_containment(det[:4], k[:4])
                if iou >= MERGE_IOU_THRESHOLD or containment >= MERGE_CONTAINMENT_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(det)
        detections.extend(kept)

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


def calculate_containment(boxA, boxB):
    """
    Intersection over smaller-box area.
    Helps remove nested duplicates where IoU alone is too low.
    """
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
    min_area = max(1e-6, min(area_a, area_b))
    return inter_area / min_area


def score_frame(gt_boxes, pred_boxes, match_threshold=0.4):
    """
    For each GT box, find the prediction with the highest IoU.
    A GT box is a missed detection if best IoU <= match_threshold.
    Returns mean IoU, missed detection count, and near-miss count
    where best IoU is in [0.3, 0.5).
    """
    if not gt_boxes:
        return 0.0, 0, 0

    best_ious = []
    missed_detections = 0
    near_misses = 0

    for gt in gt_boxes:
        gt_box = gt["box"]
        best_iou = 0.0

        for pred in pred_boxes:
            iou = calculate_iou(gt_box, pred[:4])
            if iou > best_iou:
                best_iou = iou

        best_ious.append(best_iou)
        if best_iou <= match_threshold:
            missed_detections += 1
        if 0.3 <= best_iou < 0.5:
            near_misses += 1

    mean_iou = float(np.mean(best_ious))
    return mean_iou, missed_detections, near_misses


def detection_metrics(gt_boxes, pred_boxes, match_threshold=0.4):
    """
    Greedy one-to-one matching using IoU threshold.
    Returns (tp, fp, fn, precision, recall, f1).
    """
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, 0.0, 0.0, 0.0

    matched_pred_indices = set()
    tp = 0

    for gt in gt_boxes:
        gt_box = gt["box"]
        best_iou = 0.0
        best_pred_idx = -1
        for pred_idx, pred in enumerate(pred_boxes):
            if pred_idx in matched_pred_indices:
                continue
            iou = calculate_iou(gt_box, pred[:4])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx != -1 and best_iou >= match_threshold:
            matched_pred_indices.add(best_pred_idx)
            tp += 1

    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


if __name__ == "__main__":
    # 3. Loop through first 5 samples of the first scene
    scene = nusc.scene[0]
    sample_token = scene["first_sample_token"]
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    total_tp, total_fp, total_fn = 0, 0, 0

    for idx in range(5):
        if not sample_token:
            break

        sample = nusc.get("sample", sample_token)
        cam_front_token = sample["data"]["CAM_FRONT"]
        cam_front_data = nusc.get("sample_data", cam_front_token)
        image_path = nusc.get_sample_data_path(cam_front_token)

        gt_boxes = get_2d_ground_truth(nusc, cam_front_token)
        yolo_boxes = get_yolo_predictions(image_path)
        mean_iou, missed, near_misses = score_frame(gt_boxes, yolo_boxes, match_threshold=0.4)
        tp, fp, fn, precision, recall, f1 = detection_metrics(gt_boxes, yolo_boxes, match_threshold=0.4)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"Sample {idx + 1} ({cam_front_data['filename']}): "
            f"GT boxes = {len(gt_boxes)} | YOLO detections = {len(yolo_boxes)} | "
            f"Mean IoU = {mean_iou:.3f} | Missed Detections = {missed} | "
            f"Near Misses (0.3-0.5 IoU) = {near_misses} | "
            f"TP = {tp} | FP = {fp} | FN = {fn} | "
            f"Precision = {precision:.3f} | Recall = {recall:.3f} | F1 = {f1:.3f}"
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
            xmin, ymin, xmax, ymax, conf, cls_name = pred
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
                f"Pred {cls_name} {conf:.2f}",
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

        output_path = os.path.join(output_dir, f"output_sample_{idx + 1}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        sample_token = sample["next"]

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2.0 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )
    print(
        f"Overall (5 samples): TP = {total_tp} | FP = {total_fp} | FN = {total_fn} | "
        f"Precision = {overall_precision:.3f} | Recall = {overall_recall:.3f} | F1 = {overall_f1:.3f}"
    )