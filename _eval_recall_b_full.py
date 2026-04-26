import numpy as np
from nuscenes.utils.geometry_utils import view_points
import task1

IOU_F1_THRESHOLD = 0.5
VEHICLE_CLASSES = ['vehicle.car', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck']

# recall_b candidate
cfg = {"model_conf":0.32, "global_min":0.36, "far_conf":0.54, "min_w":6.0, "min_h":6.0, "road_ratio":0.47, "merge_iou":0.39, "merge_cont":0.68}

task1.sahi_detection_model.confidence_threshold = cfg["model_conf"]
task1.GLOBAL_MIN_CONFIDENCE = cfg["global_min"]
task1.FAR_OBJECT_MIN_CONFIDENCE = cfg["far_conf"]
task1.MIN_BOX_WIDTH_PX = cfg["min_w"]
task1.MIN_BOX_HEIGHT_PX = cfg["min_h"]
task1.ROAD_ROI_START_RATIO = cfg["road_ratio"]
task1.MERGE_IOU_THRESHOLD = cfg["merge_iou"]
task1.MERGE_CONTAINMENT_THRESHOLD = cfg["merge_cont"]

def iou2d(a,b):
    xA=max(a[0],b[0]); yA=max(a[1],b[1]); xB=min(a[2],b[2]); yB=min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    areaA=max(0,a[2]-a[0])*max(0,a[3]-a[1]); areaB=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/float(areaA+areaB-inter+1e-6)

def gt_2d(tok):
    cam_data = task1.nusc.get('sample_data', tok)
    K = np.array(task1.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic'])
    _, gt3d, _ = task1.nusc.get_sample_data(tok)
    out=[]
    for g in gt3d:
        if any(v in g.name for v in VEHICLE_CLASSES):
            c2d = view_points(g.corners(), K, normalize=True)[:2,:]
            xmin=np.max([0,np.min(c2d[0])]); ymin=np.max([0,np.min(c2d[1])])
            xmax=np.min([1600,np.max(c2d[0])]); ymax=np.min([900,np.max(c2d[1])])
            if xmax>xmin and ymax>ymin:
                out.append((float(xmin),float(ymin),float(xmax),float(ymax)))
    return out

cam_front=[sd['token'] for sd in task1.nusc.sample_data if sd.get('is_key_frame',False) and sd.get('channel','')=='CAM_FRONT']
print(f"full_cam_front_samples={len(cam_front)}", flush=True)

all_iou=[]; TP=FP=FN=0
for idx,tok in enumerate(cam_front, start=1):
    gt=gt_2d(tok)
    pred=task1.get_yolo_predictions(task1.nusc.get_sample_data_path(tok))
    matched=set()
    for p in sorted(pred,key=lambda x:x[4], reverse=True):
        best_iou=0.0; best_idx=-1
        for i,g in enumerate(gt):
            if i in matched: continue
            v=iou2d(p[:4],g)
            if v>best_iou:
                best_iou=v; best_idx=i
        if best_iou>0:
            matched.add(best_idx)
            all_iou.append(best_iou)
            if best_iou>=IOU_F1_THRESHOLD:
                TP+=1
            else:
                FP+=1; FN+=1
        else:
            all_iou.append(0.0); FP+=1
    for i in range(len(gt)):
        if i not in matched:
            all_iou.append(0.0); FN+=1
    if idx % 20 == 0 or idx == len(cam_front):
        print(f"progress {idx}/{len(cam_front)}", flush=True)

miou=float(np.mean(all_iou)) if all_iou else 0.0
p=TP/(TP+FP) if (TP+FP) else 0.0
r=TP/(TP+FN) if (TP+FN) else 0.0
f1=(2*p*r/(p+r)) if (p+r) else 0.0
print(f"RESULT recall_b mIoU={miou:.6f} F1@0.5={f1:.6f} TP={TP} FP={FP} FN={FN} P={p:.6f} R={r:.6f}", flush=True)
