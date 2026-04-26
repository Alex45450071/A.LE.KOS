import json
from collections import defaultdict

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

# --- CONFIGURATION ---
DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
VERSION = "v1.0-eval"
PREDICTIONS_FILE = "predictions.json"
DT = 0.5


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _motion_label(past_xy):
    """Classify motion from past trajectory: stationary / straight / turning / uncertain."""
    if len(past_xy) < 2:
        return "uncertain"
    deltas = np.diff(past_xy[:, :2], axis=0)
    dists = np.linalg.norm(deltas, axis=1)
    speeds = dists / DT
    speed = float(np.median(speeds)) if len(speeds) else 0.0
    valid = dists > 1e-3
    if not np.any(valid):
        return "stationary"
    headings = np.arctan2(deltas[valid, 1], deltas[valid, 0])
    if len(headings) >= 2:
        yaw_diffs = np.array(
            [_wrap_angle(headings[i] - headings[i - 1]) for i in range(1, len(headings))],
            dtype=float,
        )
        yaw_rate = float(np.median(yaw_diffs) / DT) if len(yaw_diffs) else 0.0
    else:
        yaw_rate = 0.0

    if speed < 0.6:
        return "stationary"
    if abs(yaw_rate) > 0.12:
        return "turning"
    if speed > 0.8:
        return "straight"
    return "uncertain"


def _summ(values):
    if len(values) == 0:
        return {"n": 0}
    arr = np.array(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _fmt(name, stats):
    if stats.get("n", 0) == 0:
        return f"{name:<20} n=0"
    return (
        f"{name:<20} n={stats['n']:<5} "
        f"mean={stats['mean']:.3f} med={stats['median']:.3f} "
        f"p90={stats['p90']:.3f} p95={stats['p95']:.3f}"
    )


def main():
    print("Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    helper = PredictHelper(nusc)

    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    all_ade = []
    all_fde = []
    skipped_no_future = 0

    horizon_ade = defaultdict(list)
    horizon_fde = defaultdict(list)
    motion_ade = defaultdict(list)
    motion_fde = defaultdict(list)

    print(f"Evaluating {len(predictions)} predictions...")
    for i, pred in enumerate(predictions):
        sample_token = pred["sample_token"]
        instance_token = pred["instance_token"]
        pred_traj = np.array(pred["trajectories"], dtype=float)

        gt_future = helper.get_future_for_agent(
            instance_token, sample_token, seconds=6, in_agent_frame=False
        )
        if len(gt_future) == 0:
            skipped_no_future += 1
            continue

        num_steps = min(len(pred_traj), len(gt_future))
        pred_segment = pred_traj[:num_steps]
        gt_segment = gt_future[:num_steps, :2]
        distances = np.linalg.norm(pred_segment - gt_segment, axis=1)
        ade = float(np.mean(distances))
        fde = float(distances[-1])

        # For mode slicing, use last 2 seconds of past
        past = helper.get_past_for_agent(
            instance_token, sample_token, seconds=2, in_agent_frame=False
        )
        past_xy = np.array(past, dtype=float) if len(past) else np.empty((0, 2), dtype=float)
        mode = _motion_label(past_xy)

        all_ade.append(ade)
        all_fde.append(fde)
        horizon_ade[num_steps].append(ade)
        horizon_fde[num_steps].append(fde)
        motion_ade[mode].append(ade)
        motion_fde[mode].append(fde)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(predictions)}...")

    print("\n=== Overall ===")
    print(_fmt("ADE", _summ(all_ade)))
    print(_fmt("FDE", _summ(all_fde)))
    print(f"Evaluated={len(all_ade)} skipped_no_future={skipped_no_future}")

    print("\n=== By Horizon (steps compared) ===")
    for k in sorted(horizon_ade.keys()):
        print(_fmt(f"ADE@{k:02d}", _summ(horizon_ade[k])))
    print("---")
    for k in sorted(horizon_fde.keys()):
        print(_fmt(f"FDE@{k:02d}", _summ(horizon_fde[k])))

    print("\n=== By Motion Mode ===")
    for mode in ("stationary", "straight", "turning", "uncertain"):
        print(_fmt(f"ADE {mode}", _summ(motion_ade[mode])))
    print("---")
    for mode in ("stationary", "straight", "turning", "uncertain"):
        print(_fmt(f"FDE {mode}", _summ(motion_fde[mode])))


if __name__ == "__main__":
    main()
