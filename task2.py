import os.path as osp
import os
import json
import numpy as np
from sklearn.cluster import DBSCAN
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import task1

# 1. Initialize NuScenes
DATAROOT = r"C:\Users\alex_\Desktop\ALEKOS\student_dataset"
nusc = NuScenes(version="v1.0-eval", dataroot=DATAROOT, verbose=False)
MIN_CLUSTER_POINTS = 10
TRIM_PERCENTILE_LOW = 10
TRIM_PERCENTILE_HIGH = 90
DEPTH_TRIM_LOW = 5
DEPTH_TRIM_HIGH = 95
TASK2_CAR_MIN_CONF = 0.35
TASK2_CAR_NMS_IOU = 0.45
TASK2_MAX_BOX_AREA_RATIO = 0.40
DBSCAN_EPS = 0.8
DBSCAN_MIN_SAMPLES = 6
ENABLE_DBSCAN_CLEANUP = False


def _map_pointcloud_to_image(nusc, pointsensor_token, camera_token, min_dist=1.0):
    """
    Project LiDAR points into camera image and return:
    - points_2d: np.ndarray shape (2, N)
    - points_3d_cam: np.ndarray shape (3, N) in camera sensor frame
    """
    cam_sd = nusc.get("sample_data", camera_token)
    lidar_sd = nusc.get("sample_data", pointsensor_token)

    pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, lidar_sd["filename"]))

    # 1) LiDAR sensor -> ego (LiDAR timestamp)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(lidar_cs["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_cs["translation"]))

    # 2) Ego -> global
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    pc.rotate(Quaternion(lidar_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_pose["translation"]))

    # 3) Global -> ego (camera timestamp)
    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    pc.translate(-np.array(cam_pose["translation"]))
    pc.rotate(Quaternion(cam_pose["rotation"]).rotation_matrix.T)

    # 4) Ego -> camera sensor
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    pc.translate(-np.array(cam_cs["translation"]))
    pc.rotate(Quaternion(cam_cs["rotation"]).rotation_matrix.T)

    points_3d_cam = pc.points[:3, :]
    depths = points_3d_cam[2, :]
    intrinsics = np.array(cam_cs["camera_intrinsic"])
    points_2d = view_points(points_3d_cam, intrinsics, normalize=True)[:2, :]

    image_w, image_h = cam_sd["width"], cam_sd["height"]
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points_2d[0, :] >= 0)
    mask = np.logical_and(mask, points_2d[0, :] < image_w)
    mask = np.logical_and(mask, points_2d[1, :] >= 0)
    mask = np.logical_and(mask, points_2d[1, :] < image_h)

    return points_2d[:, mask], points_3d_cam[:, mask]


def get_lidar_points_in_2d_box(sample_data_token, yolo_boxes):
    """
    Args:
        sample_data_token: camera sample_data token (e.g. CAM_FRONT token)
        yolo_boxes: list of boxes in [xmin, ymin, xmax, ymax] format
                    or list of dicts containing key 'box'

    Returns:
        dict mapping each 2D box tuple -> np.ndarray of 3D points (N, 3)
        in camera sensor coordinates.
    """
    cam_sd = nusc.get("sample_data", sample_data_token)
    sample = nusc.get("sample", cam_sd["sample_token"])
    lidar_token = sample["data"]["LIDAR_TOP"]

    # Required retrieval step (also validates token exists and is readable).
    nusc.get_sample_data(lidar_token)

    points_2d, points_3d_cam = _map_pointcloud_to_image(nusc, lidar_token, sample_data_token)

    clusters_by_box = {}
    for box_item in yolo_boxes:
        box = box_item["box"] if isinstance(box_item, dict) else box_item
        xmin, ymin, xmax, ymax = map(float, box)

        inside = (
            (points_2d[0, :] >= xmin)
            & (points_2d[0, :] <= xmax)
            & (points_2d[1, :] >= ymin)
            & (points_2d[1, :] <= ymax)
        )

        # Shape: (N_points, 3), camera sensor frame coordinates.
        clusters_by_box[(xmin, ymin, xmax, ymax)] = points_3d_cam[:, inside].T

    return clusters_by_box


def get_lidar_points_for_task1_predictions(sample_data_token):
    """
    Convenience wrapper:
    - runs Task 1 YOLO detector on the camera image for sample_data_token
    - extracts 2D boxes
    - returns LiDAR clusters per predicted 2D box
    """
    image_path = nusc.get_sample_data_path(sample_data_token)
    task1_detections = task1.get_yolo_predictions(image_path)
    yolo_boxes = [det[:4] for det in task1_detections]
    return get_lidar_points_in_2d_box(sample_data_token, yolo_boxes)


def _nms_boxes(detections, iou_threshold=TASK2_CAR_NMS_IOU):
    """Simple confidence-sorted NMS for detections in [x1,y1,x2,y2,conf,class]."""
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    for det in dets:
        if all(task1.calculate_iou(det[:4], k[:4]) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def get_refined_task2_car_detections(sample_data_token):
    """
    Build cleaner 2D car boxes for Task 2 matching/3D lifting.
    - keep only car class
    - apply confidence floor
    - suppress duplicate boxes with NMS
    - drop unrealistically huge boxes (likely loose/merged detections)
    """
    image_path = nusc.get_sample_data_path(sample_data_token)
    detections = task1.get_yolo_predictions(image_path)
    cam_sd = nusc.get("sample_data", sample_data_token)
    image_area = float(cam_sd["width"] * cam_sd["height"])

    car_dets = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_name = det
        if cls_name != "car":
            continue
        if conf < TASK2_CAR_MIN_CONF:
            continue
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if box_area / image_area > TASK2_MAX_BOX_AREA_RATIO:
            continue
        car_dets.append(det)

    return _nms_boxes(car_dets, iou_threshold=TASK2_CAR_NMS_IOU)


def filter_clusters_by_size(clusters_by_box, min_points=MIN_CLUSTER_POINTS):
    """Drop tiny/noisy clusters that are unlikely to be useful for 3D fitting."""
    return {box: pts for box, pts in clusters_by_box.items() if pts.shape[0] >= min_points}


def clean_cluster_with_dbscan(cluster_points, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """
    Denoise a frustum cluster by keeping the largest DBSCAN non-noise component.
    """
    if cluster_points.shape[0] < min_samples:
        return cluster_points

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cluster_points)
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return cluster_points

    unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
    largest_label = unique_labels[np.argmax(counts)]
    return cluster_points[labels == largest_label]


def apply_dbscan_cleanup(clusters_by_box, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """Apply DBSCAN denoising to every box cluster."""
    if not ENABLE_DBSCAN_CLEANUP:
        return clusters_by_box
    return {box: clean_cluster_with_dbscan(pts, eps=eps, min_samples=min_samples) for box, pts in clusters_by_box.items()}


def serialize_clusters(clusters_by_box):
    serializable = {}
    for box, pts in clusters_by_box.items():
        key = ",".join([f"{v:.3f}" for v in box])
        serializable[key] = {
            "num_points": int(pts.shape[0]),
            "points_xyz_camera_frame": pts.tolist(),
        }
    return serializable


def estimate_robust_center(cluster_points, low_pct=TRIM_PERCENTILE_LOW, high_pct=TRIM_PERCENTILE_HIGH):
    """
    Robust center estimator:
    - trims outlier points per-axis using percentile bounds
    - returns median XYZ of the inlier subset
    """
    if cluster_points.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    low = np.percentile(cluster_points, low_pct, axis=0)
    high = np.percentile(cluster_points, high_pct, axis=0)
    inlier_mask = np.all((cluster_points >= low) & (cluster_points <= high), axis=1)
    inliers = cluster_points[inlier_mask]
    if inliers.shape[0] == 0:
        inliers = cluster_points
    return np.median(inliers, axis=0)


def estimate_depth_trimmed_center(cluster_points, low_pct=DEPTH_TRIM_LOW, high_pct=DEPTH_TRIM_HIGH):
    """
    Trim only by depth (Z in camera frame), then take XYZ median.
    Often more stable than full XYZ trimming for frustum clusters.
    """
    if cluster_points.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    z = cluster_points[:, 2]
    z_low = np.percentile(z, low_pct)
    z_high = np.percentile(z, high_pct)
    inlier_mask = (z >= z_low) & (z <= z_high)
    inliers = cluster_points[inlier_mask]
    if inliers.shape[0] == 0:
        inliers = cluster_points
    return np.median(inliers, axis=0)


def get_car_gt_2d_and_3d(sample_data_token):
    """
    Build GT car 2D boxes and GT 3D centers in camera frame.
    Returns list of dicts: {'box': [xmin, ymin, xmax, ymax], 'center': np.ndarray(3,)}.
    """
    cam_sd = nusc.get("sample_data", sample_data_token)
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_intrinsic = np.array(cam_cs["camera_intrinsic"])
    image_w, image_h = cam_sd["width"], cam_sd["height"]

    _, boxes_3d, _ = nusc.get_sample_data(sample_data_token)
    gt = []
    for box in boxes_3d:
        if box.name != "vehicle.car":
            continue
        corners_2d = view_points(box.corners(), cam_intrinsic, normalize=True)[:2, :]
        xmin, ymin = np.min(corners_2d, axis=1)
        xmax, ymax = np.max(corners_2d, axis=1)
        xmin = max(0.0, float(xmin))
        ymin = max(0.0, float(ymin))
        xmax = min(float(image_w), float(xmax))
        ymax = min(float(image_h), float(ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        gt.append({"box": [xmin, ymin, xmax, ymax], "center": np.array(box.center, dtype=float)})
    return gt


def evaluate_task2_mean_center_error(num_samples=5, iou_threshold=0.5, min_cluster_points=MIN_CLUSTER_POINTS):
    """
    Task 2 metric proxy:
    - Match predicted 2D car boxes to GT 2D car boxes by best IoU (>= iou_threshold), one-to-one.
    - For matched pairs, compute Euclidean error between predicted 3D center and GT 3D center.
    - Predicted 3D center = median XYZ of LiDAR points inside predicted 2D box.
    Returns summary dict including:
    - baseline median
    - XYZ-trim robust median
    - depth-only (Z-trim) median
    """
    scene = nusc.scene[0]
    sample_token = scene["first_sample_token"]
    all_errors_baseline = []
    all_errors_robust = []
    all_errors_depth_trim = []
    total_matches = 0

    for idx in range(num_samples):
        if not sample_token:
            break
        sample = nusc.get("sample", sample_token)
        cam_front_token = sample["data"]["CAM_FRONT"]
        image_path = nusc.get_sample_data_path(cam_front_token)

        pred_cars = get_refined_task2_car_detections(cam_front_token)
        pred_boxes = [det[:4] for det in pred_cars]
        clusters_by_box = get_lidar_points_in_2d_box(cam_front_token, pred_boxes)
        clusters_by_box = apply_dbscan_cleanup(clusters_by_box)
        gt_cars = get_car_gt_2d_and_3d(cam_front_token)

        matched_gt = set()
        sample_errors_baseline = []
        sample_errors_robust = []
        sample_errors_depth_trim = []

        for pred in sorted(pred_cars, key=lambda d: d[4], reverse=True):
            pred_box = pred[:4]
            pred_key = tuple(map(float, pred_box))
            cluster = clusters_by_box.get(pred_key)
            if cluster is None or cluster.shape[0] < min_cluster_points:
                continue

            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt in enumerate(gt_cars):
                if gt_idx in matched_gt:
                    continue
                iou = task1.calculate_iou(pred_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx is None or best_iou < iou_threshold:
                continue

            pred_center_baseline = np.median(cluster, axis=0)
            pred_center_robust = estimate_robust_center(cluster)
            pred_center_depth_trim = estimate_depth_trimmed_center(cluster)
            gt_center = gt_cars[best_gt_idx]["center"]
            error_baseline = float(np.linalg.norm(pred_center_baseline - gt_center))
            error_robust = float(np.linalg.norm(pred_center_robust - gt_center))
            error_depth_trim = float(np.linalg.norm(pred_center_depth_trim - gt_center))
            sample_errors_baseline.append(error_baseline)
            sample_errors_robust.append(error_robust)
            sample_errors_depth_trim.append(error_depth_trim)
            all_errors_baseline.append(error_baseline)
            all_errors_robust.append(error_robust)
            all_errors_depth_trim.append(error_depth_trim)
            matched_gt.add(best_gt_idx)

        total_matches += len(sample_errors_robust)
        sample_mce_baseline = float(np.mean(sample_errors_baseline)) if sample_errors_baseline else float("nan")
        sample_mce_robust = float(np.mean(sample_errors_robust)) if sample_errors_robust else float("nan")
        sample_mce_depth_trim = float(np.mean(sample_errors_depth_trim)) if sample_errors_depth_trim else float("nan")
        print(
            f"Task2 Eval Sample {idx + 1}: matched_cars={len(sample_errors_robust)} | "
            f"mce_median={sample_mce_baseline:.3f} m | "
            f"mce_xyz_trim={sample_mce_robust:.3f} m | "
            f"mce_depth_trim={sample_mce_depth_trim:.3f} m"
        )

        sample_token = sample["next"]

    overall_mce_baseline = float(np.mean(all_errors_baseline)) if all_errors_baseline else float("nan")
    overall_mce_robust = float(np.mean(all_errors_robust)) if all_errors_robust else float("nan")
    overall_mce_depth_trim = float(np.mean(all_errors_depth_trim)) if all_errors_depth_trim else float("nan")
    summary = {
        "samples_evaluated": num_samples,
        "matched_cars_total": total_matches,
        "dbscan_eps": DBSCAN_EPS,
        "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
        "trim_percentile_low": TRIM_PERCENTILE_LOW,
        "trim_percentile_high": TRIM_PERCENTILE_HIGH,
        "depth_trim_low": DEPTH_TRIM_LOW,
        "depth_trim_high": DEPTH_TRIM_HIGH,
        "mean_center_error_median_m": overall_mce_baseline,
        "mean_center_error_robust_m": overall_mce_robust,
        "mean_center_error_depth_trim_m": overall_mce_depth_trim,
        "all_center_errors_median_m": all_errors_baseline,
        "all_center_errors_robust_m": all_errors_robust,
        "all_center_errors_depth_trim_m": all_errors_depth_trim,
    }
    print(
        f"Task2 Overall: matched_cars={summary['matched_cars_total']} | "
        f"mce_median={summary['mean_center_error_median_m']:.3f} m | "
        f"mce_xyz_trim={summary['mean_center_error_robust_m']:.3f} m | "
        f"mce_depth_trim={summary['mean_center_error_depth_trim_m']:.3f} m"
    )
    return summary


if __name__ == "__main__":
    # Batch export clusters for first 5 samples in first scene.
    output_dir = "outputs_task2"
    os.makedirs(output_dir, exist_ok=True)
    scene = nusc.scene[0]
    sample_token = scene["first_sample_token"]

    summary = {
        "min_cluster_points_filter": MIN_CLUSTER_POINTS,
        "dbscan_eps": DBSCAN_EPS,
        "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
        "processed_samples": [],
        "totals": {
            "samples": 0,
            "raw_boxes": 0,
            "kept_boxes": 0,
            "raw_points": 0,
            "dbscan_points": 0,
            "kept_points": 0,
        },
    }

    for idx in range(5):
        if not sample_token:
            break

        sample = nusc.get("sample", sample_token)
        cam_front_token = sample["data"]["CAM_FRONT"]

        raw_clusters = get_lidar_points_for_task1_predictions(cam_front_token)
        dbscan_clusters = apply_dbscan_cleanup(raw_clusters)
        filtered_clusters = filter_clusters_by_size(dbscan_clusters, min_points=MIN_CLUSTER_POINTS)

        raw_box_count = len(raw_clusters)
        kept_box_count = len(filtered_clusters)
        raw_point_count = int(sum(pts.shape[0] for pts in raw_clusters.values()))
        dbscan_point_count = int(sum(pts.shape[0] for pts in dbscan_clusters.values()))
        kept_point_count = int(sum(pts.shape[0] for pts in filtered_clusters.values()))

        file_name = f"lidar_clusters_sample_{idx + 1}.json"
        output_path = osp.join(output_dir, file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialize_clusters(filtered_clusters), f, indent=2)

        summary["processed_samples"].append(
            {
                "sample_index": idx + 1,
                "cam_front_token": cam_front_token,
                "output_file": file_name,
                "raw_boxes": raw_box_count,
                "kept_boxes": kept_box_count,
                "raw_points": raw_point_count,
                "dbscan_points": dbscan_point_count,
                "kept_points": kept_point_count,
            }
        )
        summary["totals"]["samples"] += 1
        summary["totals"]["raw_boxes"] += raw_box_count
        summary["totals"]["kept_boxes"] += kept_box_count
        summary["totals"]["raw_points"] += raw_point_count
        summary["totals"]["dbscan_points"] += dbscan_point_count
        summary["totals"]["kept_points"] += kept_point_count

        print(
            f"Sample {idx + 1}: raw_boxes={raw_box_count}, kept_boxes={kept_box_count}, "
            f"raw_points={raw_point_count}, dbscan_points={dbscan_point_count}, kept_points={kept_point_count}"
        )
        print(f"Saved: {output_path}")

        sample_token = sample["next"]

    summary_path = osp.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    task2_eval = evaluate_task2_mean_center_error(num_samples=5, iou_threshold=0.5)
    eval_path = osp.join(output_dir, "task2_center_error_eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(task2_eval, f, indent=2)
    print(f"Saved Task 2 center-error eval: {eval_path}")