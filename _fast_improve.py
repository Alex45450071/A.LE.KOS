import task1

# Fast search around current two best candidates.
CONFIGS = [
    {"name":"combo_1", "model_conf":0.35, "global_min":0.41, "far_conf":0.58, "min_w":8.0, "min_h":8.0, "road_ratio":0.48, "merge_iou":0.37, "merge_cont":0.64},
    {"name":"combo_1_relaxed_far", "model_conf":0.35, "global_min":0.41, "far_conf":0.54, "min_w":8.0, "min_h":8.0, "road_ratio":0.48, "merge_iou":0.37, "merge_cont":0.64},
    {"name":"combo_1_more_recall", "model_conf":0.34, "global_min":0.39, "far_conf":0.54, "min_w":8.0, "min_h":8.0, "road_ratio":0.47, "merge_iou":0.38, "merge_cont":0.66},
    {"name":"combo_1_less_fp", "model_conf":0.36, "global_min":0.43, "far_conf":0.58, "min_w":10.0, "min_h":10.0, "road_ratio":0.49, "merge_iou":0.36, "merge_cont":0.62},
]

SEARCH_IMAGES = 20
VALIDATE_IMAGES = 40

cam_tokens = [
    sd["token"] for sd in task1.nusc.sample_data
    if sd.get("is_key_frame", False) and sd.get("channel", "").startswith("CAM_")
]
search_tokens = cam_tokens[:SEARCH_IMAGES]
validate_tokens = cam_tokens[:VALIDATE_IMAGES]


def apply(cfg):
    task1.sahi_detection_model.confidence_threshold = cfg["model_conf"]
    task1.GLOBAL_MIN_CONFIDENCE = cfg["global_min"]
    task1.FAR_OBJECT_MIN_CONFIDENCE = cfg["far_conf"]
    task1.MIN_BOX_WIDTH_PX = cfg["min_w"]
    task1.MIN_BOX_HEIGHT_PX = cfg["min_h"]
    task1.ROAD_ROI_START_RATIO = cfg["road_ratio"]
    task1.MERGE_IOU_THRESHOLD = cfg["merge_iou"]
    task1.MERGE_CONTAINMENT_THRESHOLD = cfg["merge_cont"]


def eval_cfg(cfg, tokens):
    apply(cfg)
    tp = fp = fn = 0
    for tok in tokens:
        img = task1.nusc.get_sample_data_path(tok)
        gt = task1.get_2d_ground_truth(task1.nusc, tok)
        pred = task1.get_yolo_predictions(img)
        tpi, fpi, fni, _, _, _ = task1.detection_metrics(gt, pred, match_threshold=0.4)
        tp += tpi
        fp += fpi
        fn += fni
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return tp, fp, fn, p, r, f1

print(f"SEARCH on {SEARCH_IMAGES} images")
search_results = []
for cfg in CONFIGS:
    tp, fp, fn, p, r, f1 = eval_cfg(cfg, search_tokens)
    search_results.append((f1, cfg, tp, fp, fn, p, r))
    print(f"SEARCH {cfg['name']} F1={f1:.6f} P={p:.6f} R={r:.6f} TP={tp} FP={fp} FN={fn}")

search_results.sort(key=lambda x: x[0], reverse=True)
best_cfg = search_results[0][1]
print(f"BEST_ON_SEARCH {best_cfg['name']}")

print(f"VALIDATE on {VALIDATE_IMAGES} images")
for tag, cfg in [("winner", best_cfg), ("baseline_combo_1", CONFIGS[0])]:
    tp, fp, fn, p, r, f1 = eval_cfg(cfg, validate_tokens)
    print(f"VALIDATE {tag} {cfg['name']} F1={f1:.6f} P={p:.6f} R={r:.6f} TP={tp} FP={fp} FN={fn}")
