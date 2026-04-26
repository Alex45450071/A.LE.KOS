import random
import task1

NUM_RANDOM_TESTS = 6
SLICE_SIZE = 10
BASE_SEED = 2026


def apply_static_combo_1():
    task1.sahi_detection_model.confidence_threshold = 0.35
    task1.GLOBAL_MIN_CONFIDENCE = 0.41
    task1.FAR_OBJECT_MIN_CONFIDENCE = 0.58
    task1.MIN_BOX_WIDTH_PX = 8.0
    task1.MIN_BOX_HEIGHT_PX = 8.0
    task1.ROAD_ROI_START_RATIO = 0.48
    task1.MERGE_IOU_THRESHOLD = 0.37
    task1.MERGE_CONTAINMENT_THRESHOLD = 0.64


def run_sequence(tokens, temporal_enabled):
    task1.AUTO_DYNAMIC_PROFILE = True
    if temporal_enabled:
        temporal_input = []
        gt_per_frame = []
        for tok in tokens:
            image_path = task1.nusc.get_sample_data_path(tok)
            gt_boxes = task1.get_2d_ground_truth(task1.nusc, tok)
            pred = task1.get_yolo_predictions(image_path)
            temporal_input.append(pred)
            gt_per_frame.append(gt_boxes)

        boosted = task1.apply_temporal_consistency_boost(temporal_input)

        tp = fp = fn = 0
        for gt_boxes, pred_boxes in zip(gt_per_frame, boosted):
            tpi, fpi, fni, _, _, _ = task1.detection_metrics(gt_boxes, pred_boxes, match_threshold=0.4)
            tp += tpi; fp += fpi; fn += fni
    else:
        apply_static_combo_1()
        tp = fp = fn = 0
        for tok in tokens:
            image_path = task1.nusc.get_sample_data_path(tok)
            gt_boxes = task1.get_2d_ground_truth(task1.nusc, tok)
            pred_boxes = task1.get_yolo_predictions(image_path)
            tpi, fpi, fni, _, _, _ = task1.detection_metrics(gt_boxes, pred_boxes, match_threshold=0.4)
            tp += tpi; fp += fpi; fn += fni

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return tp, fp, fn, p, r, f1


all_cam_tokens = [
    sd["token"]
    for sd in task1.nusc.sample_data
    if sd.get("is_key_frame", False) and sd.get("channel", "").startswith("CAM_")
]

max_start = max(0, len(all_cam_tokens) - SLICE_SIZE)
print(f"RANDOM_A/B tests={NUM_RANDOM_TESTS} slice={SLICE_SIZE} seed={BASE_SEED}")

wins_temporal = 0
wins_static = 0
ties = 0
sum_delta = 0.0

for i in range(NUM_RANDOM_TESTS):
    rng = random.Random(BASE_SEED + i)
    start = rng.randint(0, max_start)
    tokens = all_cam_tokens[start : start + SLICE_SIZE]

    t_tp, t_fp, t_fn, t_p, t_r, t_f1 = run_sequence(tokens, temporal_enabled=True)
    s_tp, s_fp, s_fn, s_p, s_r, s_f1 = run_sequence(tokens, temporal_enabled=False)

    delta = t_f1 - s_f1
    sum_delta += delta

    if delta > 1e-9:
        wins_temporal += 1
    elif delta < -1e-9:
        wins_static += 1
    else:
        ties += 1

    print(
        f"TEST {i+1} start={start} temporal_F1={t_f1:.6f} static_F1={s_f1:.6f} delta={delta:+.6f} "
        f"temporal(P={t_p:.6f},R={t_r:.6f},FP={t_fp}) static(P={s_p:.6f},R={s_r:.6f},FP={s_fp})"
    )

avg_delta = sum_delta / NUM_RANDOM_TESTS if NUM_RANDOM_TESTS else 0.0
print(f"SUMMARY temporal_wins={wins_temporal} static_wins={wins_static} ties={ties} avg_delta={avg_delta:+.6f}")
