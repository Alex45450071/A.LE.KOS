import task1

SLICE_SIZE = 10
START = 50


def apply_static_combo_1():
    task1.sahi_detection_model.confidence_threshold = 0.35
    task1.GLOBAL_MIN_CONFIDENCE = 0.41
    task1.FAR_OBJECT_MIN_CONFIDENCE = 0.58
    task1.MIN_BOX_WIDTH_PX = 8.0
    task1.MIN_BOX_HEIGHT_PX = 8.0
    task1.ROAD_ROI_START_RATIO = 0.48
    task1.MERGE_IOU_THRESHOLD = 0.37
    task1.MERGE_CONTAINMENT_THRESHOLD = 0.64


def eval_tokens(tokens, temporal_enabled):
    task1.AUTO_DYNAMIC_PROFILE = True
    tp = fp = fn = 0

    if temporal_enabled:
        seq_preds = []
        seq_gt = []
        for tok in tokens:
            img = task1.nusc.get_sample_data_path(tok)
            gt = task1.get_2d_ground_truth(task1.nusc, tok)
            pred = task1.get_yolo_predictions(img)
            seq_gt.append(gt)
            seq_preds.append(pred)
        boosted = task1.apply_temporal_consistency_boost(seq_preds)
        for gt, pred in zip(seq_gt, boosted):
            tpi, fpi, fni, _, _, _ = task1.detection_metrics(gt, pred, match_threshold=0.4)
            tp += tpi; fp += fpi; fn += fni
    else:
        apply_static_combo_1()
        for tok in tokens:
            img = task1.nusc.get_sample_data_path(tok)
            gt = task1.get_2d_ground_truth(task1.nusc, tok)
            pred = task1.get_yolo_predictions(img)
            tpi, fpi, fni, _, _, _ = task1.detection_metrics(gt, pred, match_threshold=0.4)
            tp += tpi; fp += fpi; fn += fni

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return tp, fp, fn, p, r, f1

side_tokens = [
    sd["token"]
    for sd in task1.nusc.sample_data
    if sd.get("is_key_frame", False) and sd.get("channel", "") == "CAM_LEFT"
]

slice_tokens = side_tokens[START:START + SLICE_SIZE]
print(f"SIDE_TEST channel=CAM_LEFT start={START} size={len(slice_tokens)}")

for name, temporal in [("temporal", True), ("static_combo_1", False)]:
    tp, fp, fn, p, r, f1 = eval_tokens(slice_tokens, temporal)
    print(f"RESULT {name} TP={tp} FP={fp} FN={fn} P={p:.6f} R={r:.6f} F1={f1:.6f}")
