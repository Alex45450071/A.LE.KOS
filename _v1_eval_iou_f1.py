import numpy as np
from nuscenes.utils.geometry_utils import view_points
import task1

VEHICLE_CLASSES = ['vehicle.car', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck']
IOU_F1_THRESHOLD = 0.5


def calculate_iou_2d(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def gt_2d_for_cam_token(cam_token):
    cam_data = task1.nusc.get('sample_data', cam_token)
    cam_intrinsic = np.array(task1.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic'])
    _, gt_boxes_3d, _ = task1.nusc.get_sample_data(cam_token)

    gt_2d_boxes = []
    for gt_box in gt_boxes_3d:
        if any(v_class in gt_box.name for v_class in VEHICLE_CLASSES):
            corners_2d = view_points(gt_box.corners(), cam_intrinsic, normalize=True)[:2, :]
            xmin = np.max([0, np.min(corners_2d[0])])
            ymin = np.max([0, np.min(corners_2d[1])])
            xmax = np.min([1600, np.max(corners_2d[0])])
            ymax = np.min([900, np.max(corners_2d[1])])
            if xmax > xmin and ymax > ymin:
                gt_2d_boxes.append((float(xmin), float(ymin), float(xmax), float(ymax)))
    return gt_2d_boxes


cam_front_tokens = [
    sd['token'] for sd in task1.nusc.sample_data
    if sd.get('is_key_frame', False) and sd.get('channel', '') == 'CAM_FRONT'
]

all_iou_scores = []
TP = FP = FN = 0

print(f"Evaluating v1.0-eval CAM_FRONT samples={len(cam_front_tokens)}", flush=True)

for idx, cam_token in enumerate(cam_front_tokens, start=1):
    gt_boxes = gt_2d_for_cam_token(cam_token)
    image_path = task1.nusc.get_sample_data_path(cam_token)
    pred_boxes = task1.get_yolo_predictions(image_path)

    matched_gt = set()
    preds_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

    for p in preds_sorted:
        p2d = p[:4]
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou_2d(p2d, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou > 0.0:
            matched_gt.add(best_gt_idx)
            all_iou_scores.append(best_iou)
            if best_iou >= IOU_F1_THRESHOLD:
                TP += 1
            else:
                FP += 1
                FN += 1
        else:
            all_iou_scores.append(0.0)
            FP += 1

    for gt_idx in range(len(gt_boxes)):
        if gt_idx not in matched_gt:
            all_iou_scores.append(0.0)
            FN += 1

    if idx % 20 == 0 or idx == len(cam_front_tokens):
        print(f"progress {idx}/{len(cam_front_tokens)}", flush=True)

mean_iou = float(np.mean(all_iou_scores)) if all_iou_scores else 0.0
precision = TP / (TP + FP) if (TP + FP) else 0.0
recall = TP / (TP + FN) if (TP + FN) else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

print(f"V1_EVAL_MEAN_IOU={mean_iou:.6f}", flush=True)
print(f"V1_EVAL_F1_IOU_0_5={f1:.6f}", flush=True)
print(f"TP={TP} FP={FP} FN={FN} P={precision:.6f} R={recall:.6f}", flush=True)
