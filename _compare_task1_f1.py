import json
from pathlib import Path

import numpy as np
from nuscenes.utils.geometry_utils import view_points

import task1


SUBMISSION_PATH = Path(r"c:\Users\alex_\Desktop\ALEKOS\test_submission_cam_front.json")
IOU_THRESHOLD = 0.5
TASK1_NEW_F1_50 = 0.5312


def iou_2d(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def gt_boxes_for_sample(sample_token):
    sample = task1.nusc.get("sample", sample_token)
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = task1.nusc.get("sample_data", cam_token)
    cam_intrinsic = np.array(
        task1.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])["camera_intrinsic"]
    )
    _, gt_boxes_3d, _ = task1.nusc.get_sample_data(cam_token)
    gt_2d = []
    for gt in gt_boxes_3d:
        if any(
            v_class in gt.name
            for v_class in ["vehicle.car", "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.truck"]
        ):
            corners_2d = view_points(gt.corners(), cam_intrinsic, normalize=True)[:2, :]
            xmin, ymin = np.max([0, np.min(corners_2d[0])]), np.max([0, np.min(corners_2d[1])])
            xmax, ymax = np.min([1600, np.max(corners_2d[0])]), np.min([900, np.max(corners_2d[1])])
            gt_2d.append([float(xmin), float(ymin), float(xmax), float(ymax)])
    return cam_token, gt_2d


def task1_pred_boxes(cam_token):
    image_path = task1.nusc.get_sample_data_path(cam_token)
    preds = task1.get_yolo_predictions(image_path)
    return [p[:4] for p in preds]


def evaluate_on_tokens(sample_tokens):
    tp = 0
    fp = 0
    fn = 0
    for sample_token in sample_tokens:
        cam_token, gt_2d = gt_boxes_for_sample(sample_token)
        pred_2d = task1_pred_boxes(cam_token)

        matched_gt = set()
        pred_2d = sorted(pred_2d, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        for pb in pred_2d:
            best_iou = 0.0
            best_idx = -1
            for gi, gb in enumerate(gt_2d):
                if gi in matched_gt:
                    continue
                iou = iou_2d(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_idx != -1 and best_iou >= IOU_THRESHOLD:
                matched_gt.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn += max(0, len(gt_2d) - len(matched_gt))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1


if __name__ == "__main__":
    submission = json.loads(SUBMISSION_PATH.read_text(encoding="utf-8"))
    sample_tokens = list(submission.get("detections", {}).keys())
    tp, fp, fn, precision, recall, f1 = evaluate_on_tokens(sample_tokens)
    print(
        f"task1.py on same {len(sample_tokens)} samples | "
        f"TP={tp} FP={fp} FN={fn} | Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}"
    )
    diff = f1 - TASK1_NEW_F1_50
    print(
        f"Comparison vs task1_new F1={TASK1_NEW_F1_50:.4f} | "
        f"delta(task1.py - task1_new)={diff:+.4f}"
    )
