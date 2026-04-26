import json
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from ultralytics import YOLO


DATAROOT = Path(r"c:\Users\alex_\Desktop\ALEKOS\student_dataset")
MODEL_PATH = "yolo26s.pt"
MAX_SAMPLES = 5
IOU_F1_THRESHOLD = 0.5
OUT_JSON = Path(r"c:\Users\alex_\Desktop\ALEKOS\test_submission_cam_front.json")

VEHICLE_CLASSES = {"car", "bus", "truck"}


def iou_2d(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def generate_submission():
    nusc = NuScenes(version="v1.0-eval", dataroot=str(DATAROOT), verbose=False)
    model = YOLO(MODEL_PATH)

    detections = {}
    count = 0
    for scene in nusc.scene:
        token = scene["first_sample_token"]
        while token and count < MAX_SAMPLES:
            sample = nusc.get("sample", token)
            cam_token = sample["data"]["CAM_FRONT"]
            img_path = nusc.get_sample_data_path(cam_token)
            pred = model.predict(source=img_path, verbose=False, conf=0.25, iou=0.5)[0]
            frame = []
            for box in pred.boxes:
                cls_name = model.names[int(box.cls.item())].lower()
                if cls_name not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                frame.append(
                    {
                        "class_name": cls_name,
                        "score": float(box.conf.item()),
                        "box_2d": [float(x1), float(y1), float(x2), float(y2)],
                        "box_3d_center": [0.0, 0.0, 0.0],
                        "box_3d_size": [0.0, 0.0, 0.0],
                        "box_3d_yaw": 0.0,
                    }
                )
            detections[token] = frame
            count += 1
            token = sample["next"]

    payload = {"detections": detections, "trajectories": {}}
    OUT_JSON.write_text(json.dumps(payload), encoding="utf-8")
    return nusc, payload


def run_eval_greedy_logic(nusc, preds):
    # Mirrors evaluation_greedy.py matching behavior and adds F1@0.5.
    tp = 0
    fp = 0
    fn = 0
    for sample_token, predicted_boxes in preds.get("detections", {}).items():
        try:
            sample = nusc.get("sample", sample_token)
            cam_token = sample["data"]["CAM_FRONT"]
            cam_data = nusc.get("sample_data", cam_token)
            intr = np.array(
                nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])["camera_intrinsic"]
            )
            _, gt_boxes_3d, _ = nusc.get_sample_data(cam_token)
        except Exception:
            continue

        gt_2d = []
        for gt in gt_boxes_3d:
            if any(v in gt.name for v in ["vehicle.car", "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.truck"]):
                corners_2d = view_points(gt.corners(), intr, normalize=True)[:2, :]
                xmin, ymin = np.max([0, np.min(corners_2d[0])]), np.max([0, np.min(corners_2d[1])])
                xmax, ymax = np.min([1600, np.max(corners_2d[0])]), np.min([900, np.max(corners_2d[1])])
                gt_2d.append((xmin, ymin, xmax, ymax))

        matched = set()
        predicted_boxes = sorted(predicted_boxes, key=lambda x: x.get("score", 0.0), reverse=True)
        for p in predicted_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_2d):
                if idx in matched:
                    continue
                iou = iou_2d(p["box_2d"], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx != -1 and best_iou >= IOU_F1_THRESHOLD:
                matched.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn += max(0, len(gt_2d) - len(matched))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1


if __name__ == "__main__":
    nusc_obj, pred_json = generate_submission()
    tp, fp, fn, precision, recall, f1 = run_eval_greedy_logic(nusc_obj, pred_json)
    print(f"Saved submission: {OUT_JSON}")
    print(
        f"CAM_FRONT subset={MAX_SAMPLES} | TP={tp} FP={fp} FN={fn} | "
        f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}"
    )
