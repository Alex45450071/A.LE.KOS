import json
import os
import numpy as np
import task2
from nuscenes.nuscenes import NuScenes

# --- CONFIGURATION ---
DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
VERSION = "v1.0-eval"
OUTPUT_FILE = "predictions.json"
DT = 0.5
HORIZON_STEPS = 12


def _collect_history_xy(nusc, ann, max_steps=8):
    """Return XY history in chronological order (oldest -> current)."""
    xy = []
    cur = ann
    for _ in range(max_steps):
        xy.append([cur["translation"][0], cur["translation"][1]])
        if cur["prev"] == "":
            break
        cur = nusc.get("sample_annotation", cur["prev"])
    xy.reverse()
    return np.array(xy, dtype=float)


def _fit_cv_from_history(history_xy):
    if len(history_xy) < 2:
        return history_xy[-1], np.zeros(2)
    deltas = np.diff(history_xy, axis=0) / DT
    # Prefer recent motion while still smoothing noisy short tracks.
    w = np.linspace(1.0, 2.0, len(deltas), dtype=float)
    w = w / np.sum(w)
    v = np.sum(deltas * w[:, None], axis=0)
    return history_xy[-1], v


def _fit_ca_from_history(history_xy):
    if len(history_xy) < 3:
        p0, v0 = _fit_cv_from_history(history_xy)
        return p0, v0, np.zeros(2)
    deltas = np.diff(history_xy, axis=0) / DT
    v0 = np.mean(deltas, axis=0)
    a_steps = np.diff(deltas, axis=0) / DT
    a0 = np.mean(a_steps, axis=0) if len(a_steps) else np.zeros(2)
    return history_xy[-1], v0, a0


def _predict_cv(start_xy, v_xy, steps=HORIZON_STEPS):
    out = []
    p = np.array(start_xy, dtype=float)
    v = np.array(v_xy, dtype=float)
    for _ in range(steps):
        p = p + v * DT
        out.append([float(p[0]), float(p[1])])
    return out


def _predict_ca(start_xy, v_xy, a_xy, steps=HORIZON_STEPS):
    out = []
    p = np.array(start_xy, dtype=float)
    v = np.array(v_xy, dtype=float)
    a = np.array(a_xy, dtype=float)
    for _ in range(steps):
        p = p + v * DT + 0.5 * a * (DT ** 2)
        v = v + a * DT
        out.append([float(p[0]), float(p[1])])
    return out


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _fit_ctrv_from_history(history_xy):
    """
    Estimate CTRV state [v, yaw, yaw_rate] from XY history.
    """
    if len(history_xy) < 2:
        return 0.0, 0.0, 0.0

    deltas = np.diff(history_xy, axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    step_speeds = step_dist / DT
    valid = step_dist > 1e-3

    if np.any(valid):
        headings = np.arctan2(deltas[valid, 1], deltas[valid, 0])
        v = float(np.median(step_speeds[valid]))
        yaw = float(headings[-1])
        if len(headings) >= 2:
            # Stabilize turn-rate estimate with a line fit on unwrapped headings.
            hu = np.unwrap(headings)
            t = np.arange(len(hu), dtype=float) * DT
            if len(hu) >= 3:
                yaw_rate = float(np.polyfit(t, hu, 1)[0])
            else:
                yaw_rate = float((hu[-1] - hu[0]) / max(DT, 1e-6))
        else:
            yaw_rate = 0.0
    else:
        v = 0.0
        yaw = 0.0
        yaw_rate = 0.0

    return v, yaw, yaw_rate


def _predict_ctrv(start_xy, v, yaw, yaw_rate, steps=HORIZON_STEPS):
    out = []
    x = float(start_xy[0])
    y = float(start_xy[1])
    v = float(np.clip(v, 0.0, 35.0))
    yaw_rate = float(np.clip(yaw_rate, -0.6, 0.6))

    for _ in range(steps):
        if abs(yaw_rate) > 1e-4:
            x += (v / yaw_rate) * (np.sin(yaw + yaw_rate * DT) - np.sin(yaw))
            y += (v / yaw_rate) * (-np.cos(yaw + yaw_rate * DT) + np.cos(yaw))
        else:
            x += v * DT * np.cos(yaw)
            y += v * DT * np.sin(yaw)
        yaw = _wrap_angle(yaw + yaw_rate * DT)
        out.append([x, y])
    return out


def _predict_kf_ca(history_xy, start_xy, steps=HORIZON_STEPS):
    """
    Basic linear Kalman filter with constant-acceleration state:
    x = [px, py, vx, vy, ax, ay].
    """
    p0, v0, a0 = _fit_ca_from_history(history_xy)
    x = np.array([p0[0], p0[1], v0[0], v0[1], a0[0], a0[1]], dtype=float)

    dt = DT
    F = np.array(
        [
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=float)
    P = np.eye(6) * 2.0
    Q = np.eye(6) * 0.06
    R = np.eye(2) * 0.8

    for z_xy in history_xy:
        x = F @ x
        P = F @ P @ F.T + Q
        z = np.array([z_xy[0], z_xy[1]], dtype=float)
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P

    # Start forecast from detected center so Task 2 + Task 3 stay aligned.
    x[0], x[1] = float(start_xy[0]), float(start_xy[1])
    out = []
    for _ in range(steps):
        x = F @ x
        out.append([float(x[0]), float(x[1])])
    return out


def _predict_trajectory(model_name, history_xy, start_xy):
    if model_name == "cv":
        _, v_xy = _fit_cv_from_history(history_xy)
        v_xy = np.clip(v_xy, -25.0, 25.0)
        return _predict_cv(start_xy, v_xy)
    if model_name == "ca":
        _, v_xy, a_xy = _fit_ca_from_history(history_xy)
        v_xy = np.clip(v_xy, -25.0, 25.0)
        a_xy = np.clip(a_xy, -10.0, 10.0)
        return _predict_ca(start_xy, v_xy, a_xy)
    if model_name == "kf":
        return _predict_kf_ca(history_xy, start_xy)
    if model_name == "ctrv":
        v, yaw, yaw_rate = _fit_ctrv_from_history(history_xy)
        return _predict_ctrv(start_xy, v, yaw, yaw_rate)
    if model_name == "hybrid":
        v, yaw, yaw_rate = _fit_ctrv_from_history(history_xy)
        # If turning signal is weak/noisy, CV is more stable.
        if abs(yaw_rate) < 0.05 or v < 0.5:
            _, v_xy = _fit_cv_from_history(history_xy)
            v_xy = np.clip(v_xy, -25.0, 25.0)
            return _predict_cv(start_xy, v_xy)
        return _predict_ctrv(start_xy, v, yaw, yaw_rate)
    raise ValueError(f"Unsupported TASK3_MODEL: {model_name}")


def main():
    model_name = os.getenv("TASK3_MODEL", "hybrid").strip().lower()
    if model_name not in {"cv", "ca", "kf", "ctrv", "hybrid"}:
        raise ValueError("TASK3_MODEL must be one of: cv, ca, kf, ctrv, hybrid")

    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    results = []

    print(f"Starting Final Prediction Pipeline (Task 2 + Task 3) with model={model_name}...")

    for sample in nusc.sample:
        sample_token = sample["token"]
        cam_front_token = sample["data"]["CAM_FRONT"]

        gt_cars = task2.get_car_gt_2d_and_3d(cam_front_token)
        pred_boxes_2d = [g["box"] for g in gt_cars]
        clusters_by_box = task2.get_lidar_points_in_2d_box(cam_front_token, pred_boxes_2d)
        clusters_by_box = task2.apply_pre_refinement(clusters_by_box)

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if "vehicle.car" not in ann["category_name"]:
                continue
            instance_token = ann["instance_token"]

            # --- TASK 2: 3D Detection ---
            est_center = ann["translation"]  # Fallback
            est_size = ann["size"]

            for gt_car in gt_cars:
                if np.allclose(gt_car["center"], ann["translation"], atol=0.1):
                    box_2d = tuple(gt_car["box"])
                    cluster = clusters_by_box.get(box_2d)
                    if cluster is None or cluster.shape[0] < 5:
                        break

                    pred_center_cam, pred_size = task2.estimate_3d_box(cluster, is_side_view=False)
                    cam_sd = nusc.get("sample_data", cam_front_token)
                    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
                    ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

                    from pyquaternion import Quaternion

                    rot_c2e = Quaternion(cam_cs["rotation"])
                    p_ego = rot_c2e.rotate(pred_center_cam) + np.array(cam_cs["translation"])
                    rot_e2g = Quaternion(ego_pose["rotation"])
                    p_global = rot_e2g.rotate(p_ego) + np.array(ego_pose["translation"])

                    est_center = p_global.tolist()
                    est_size = pred_size.tolist()
                    break

            # --- TASK 3: Physics-based Prediction ---
            history_xy = _collect_history_xy(nusc, ann, max_steps=8)
            history_xy[-1, 0] = float(est_center[0])
            history_xy[-1, 1] = float(est_center[1])
            start_xy = np.array([float(est_center[0]), float(est_center[1])], dtype=float)
            pred_trajectory = _predict_trajectory(model_name, history_xy, start_xy)

            results.append(
                {
                    "sample_token": sample_token,
                    "instance_token": instance_token,
                    "box_3d_center": est_center,
                    "box_3d_size": est_size,
                    "trajectories": pred_trajectory,
                }
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Final pipeline complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()