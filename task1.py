import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

# 1. Initialize NuScenes (Point this to your extracted folder)
DATAROOT = '/path/to/Participant/student_dataset'
nusc = NuScenes(version='v1.0-eval', dataroot=DATAROOT, verbose=False)

# 2. Grab a random scene and its first keyframe (sample)
my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

# 3. Get the Front Camera data
cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
cam_intrinsic = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])['camera_intrinsic']
cam_intrinsic = np.array(cam_intrinsic)

# 4. Use the devkit to load the image and the 3D boxes visible in it
data_path, boxes_3d, _ = nusc.get_sample_data(cam_front_data['token'])

# Load the image for plotting
im = Image.open(data_path)
plt.figure(figsize=(12, 8))
plt.imshow(im)

print(f"--- 2D Bounding Boxes for: {cam_front_data['filename']} ---")

# 5. Project each 3D box into a 2D box
for box in boxes_3d:

    #skipping everything that isn't a vehicle
    if not "vehicle" in box.name:
        continue

    # A 3D box has 8 corners. Get their 3D coordinates.
    corners_3d = box.corners() 
    
    # Project the 8 3D corners onto the 2D camera image using the intrinsic matrix
    corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
    
    # Find the minimum and maximum X and Y values to draw a flat 2D rectangle
    xmin, ymin = np.min(corners_2d, axis=1)
    xmax, ymax = np.max(corners_2d, axis=1)
    
    # Clip the coordinates so they don't go off the edge of the screen
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(im.size[0], xmax)
    ymax = min(im.size[1], ymax)
    
    # Print the result
    print(f"Class: {box.name}")
    print(f"2D Box [xmin, ymin, xmax, ymax]: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]\n")
    
    # Draw the box on the image
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)

plt.title("Projected 2D Ground Truth Boxes")
plt.axis('off')
plt.show()