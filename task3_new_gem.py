import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

# --- CONFIGURATION ---
DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
VERSION = "v1.0-eval"
OUTPUT_FILE = "predictions.json"
DT = 0.5
HORIZON_STEPS = 12


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _estimate_velocity_xy(history_xy):
    """Estimate XY velocity from past points in global frame."""
    if len(history_xy) < 2:
        return np.zeros(2, dtype=float)
    deltas = np.diff(history_xy[:, :2], axis=0) / DT
    # Median is more robust to noisy short histories.
    return np.median(deltas, axis=0)


def _predict_cv(start_xy, v_xy, steps=HORIZON_STEPS):
    """Constant-velocity rollout in global XY coordinates."""
    out = []
    pos = np.array(start_xy, dtype=float)
    vel = np.array(v_xy, dtype=float)
    vel = np.clip(vel, -25.0, 25.0)
    for _ in range(steps):
        pos = pos + vel * DT
        out.append([float(pos[0]), float(pos[1])])
    return out


def _fit_ctrv(history_xy):
    """Estimate speed, heading and yaw-rate from recent XY history."""
    if len(history_xy) < 2:
        return 0.0, 0.0, 0.0

    deltas = np.diff(history_xy[:, :2], axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    valid = step_dist > 1e-3
    if not np.any(valid):
        return 0.0, 0.0, 0.0

    headings = np.arctan2(deltas[valid, 1], deltas[valid, 0])
    speeds = step_dist[valid] / DT
    speed = float(np.median(speeds))
    yaw = float(headings[-1])

    if len(headings) >= 2:
        yaw_diffs = np.array([_wrap_angle(headings[i] - headings[i - 1]) for i in range(1, len(headings))], dtype=float)
        yaw_rate = float(np.median(yaw_diffs) / DT)
    else:
        yaw_rate = 0.0

    return speed, yaw, yaw_rate


def _predict_ctrv(start_xy, speed, yaw, yaw_rate, steps=HORIZON_STEPS):
    """CTRV rollout in global XY."""
    out = []
    x = float(start_xy[0])
    y = float(start_xy[1])
    speed = float(np.clip(speed, 0.0, 35.0))
    yaw_rate = float(np.clip(yaw_rate, -0.8, 0.8))

    for _ in range(steps):
        if abs(yaw_rate) > 1e-4:
            x += (speed / yaw_rate) * (np.sin(yaw + yaw_rate * DT) - np.sin(yaw))
            y += (speed / yaw_rate) * (-np.cos(yaw + yaw_rate * DT) + np.cos(yaw))
        else:
            x += speed * DT * np.cos(yaw)
            y += speed * DT * np.sin(yaw)
        yaw = _wrap_angle(yaw + yaw_rate * DT)
        out.append([x, y])
    return out


def _classify_motion_mode(history_xy):
    """
    Motion modes:
    - stationary: near-zero speed
    - turning: consistent non-trivial yaw-rate
    - straight: moving with low yaw-rate
    - uncertain: noisy/insufficient signal
    """
    if len(history_xy) < 2:
        return "uncertain"

    v_xy = _estimate_velocity_xy(history_xy)
    speed_cv = float(np.linalg.norm(v_xy))
    speed, _, yaw_rate = _fit_ctrv(history_xy)

    deltas = np.diff(history_xy[:, :2], axis=0) if len(history_xy) >= 2 else np.empty((0, 2), dtype=float)
    if len(deltas) == 0:
        return "uncertain"
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    if len(headings) >= 2:
        yaw_jitter = float(np.std([_wrap_angle(headings[i] - headings[i - 1]) for i in range(1, len(headings))]))
    else:
        yaw_jitter = 0.0

    if speed < 0.6 and speed_cv < 0.6:
        return "stationary"
    if speed > 1.5 and abs(yaw_rate) > 0.12 and yaw_jitter < 0.30:
        return "turning"
    if speed_cv > 0.8 and abs(yaw_rate) < 0.08:
        return "straight"
    return "uncertain"


def _estimate_uncertainty_scale(history_xy):
    """
    Returns uncertainty in [0, 1] from heading jitter + speed volatility.
    Higher means less reliable motion signal.
    """
    if len(history_xy) < 3:
        return 1.0
    deltas = np.diff(history_xy[:, :2], axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    speeds = step_dist / DT
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    if len(headings) >= 2:
        yaw_diffs = np.array([_wrap_angle(headings[i] - headings[i - 1]) for i in range(1, len(headings))], dtype=float)
        yaw_jitter = float(np.std(yaw_diffs))
    else:
        yaw_jitter = 0.0
    speed_vol = float(np.std(speeds))
    # Normalize to practical ranges then blend.
    yaw_score = np.clip(yaw_jitter / 0.40, 0.0, 1.0)
    speed_score = np.clip(speed_vol / 3.0, 0.0, 1.0)
    return float(0.6 * yaw_score + 0.4 * speed_score)


def _predict_hybrid(history_xy, start_xy):
    """
    Hybrid predictor:
    - CV for near-straight/slow trajectories.
    - CTRV for clear turning motion.
    - Apply mild damping over horizon to reduce long-range overshoot.
    """
    v_xy = _estimate_velocity_xy(history_xy)
    speed, yaw, yaw_rate = _fit_ctrv(history_xy)
    mode = _classify_motion_mode(history_xy)

    if mode == "stationary":
        raw = [[float(start_xy[0]), float(start_xy[1])] for _ in range(HORIZON_STEPS)]
    elif mode == "turning":
        raw = _predict_ctrv(start_xy, speed, yaw, yaw_rate, HORIZON_STEPS)
    elif mode == "straight":
        raw = _predict_cv(start_xy, v_xy, HORIZON_STEPS)
    else:
        # Conservative fallback for uncertain/noisy tracks.
        uncertainty = _estimate_uncertainty_scale(history_xy)
        conservative_scale = 0.75 - 0.45 * uncertainty
        conservative_scale = float(np.clip(conservative_scale, 0.30, 0.75))
        raw = _predict_cv(start_xy, conservative_scale * v_xy, HORIZON_STEPS)

    # Damping: blend toward constant-velocity displacement after each step.
    # This reduces exploding curvature on noisy short tracks.
    damped = []
    prev = np.array(start_xy, dtype=float)
    for i, point in enumerate(raw):
        p = np.array(point, dtype=float)
        alpha = 1.0 - 0.03 * i  # gradually stronger damping
        alpha_floor = 0.55 if mode == "uncertain" else 0.65
        alpha = float(np.clip(alpha, alpha_floor, 1.0))
        p = prev + alpha * (p - prev)
        damped.append([float(p[0]), float(p[1])])
        prev = p

    # Outlier clipping: cap per-step displacement to realistic motion.
    clipped = []
    prev = np.array(start_xy, dtype=float)
    if mode == "stationary":
        max_speed = 1.0
    elif mode == "turning":
        max_speed = 16.0
    elif mode == "straight":
        max_speed = 20.0
    else:
        max_speed = 12.0
    max_step = max_speed * DT
    for point in damped:
        p = np.array(point, dtype=float)
        d = p - prev
        dist = float(np.linalg.norm(d))
        if dist > max_step and dist > 1e-6:
            p = prev + d / dist * max_step
        clipped.append([float(p[0]), float(p[1])])
        prev = p
    return clipped


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    helper = PredictHelper(nusc)
    results = []

    print("Running Task 3 prediction with hybrid mode-classifier...")
    for sample in nusc.sample:
        sample_token = sample["token"]
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if "vehicle.car" not in ann["category_name"]:
                continue

            instance_token = ann["instance_token"]
            start_xy = np.array(ann["translation"][:2], dtype=float)
            history = helper.get_past_for_agent(
                instance_token, sample_token, seconds=2, in_agent_frame=False
            )
            history_xy = np.array(history, dtype=float) if len(history) else np.empty((0, 2), dtype=float)
            traj = _predict_hybrid(history_xy, start_xy)

            results.append(
                {
                    "sample_token": sample_token,
                    "instance_token": instance_token,
                    "trajectories": traj,
                }
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()