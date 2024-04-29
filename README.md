# rtk-challenge
50.047 Race Track Challenge Spring 2024

## Setup
Copy `rtk_challenge` folder into `src` of ROS workspace and build `rtk_challenge` package.

Source built workspace in root of workspace
```
source install/setup.bash
```

## Part A

Run the lane detection node which gets the ZED camera feed, performs lane detection, then implements a PID control loop to correct the vehicle's trajectory (Note that lane detection was disabled in the code for this part). 
```
ros2 run rtk_challenge lane_detection
```

The code for the `lane_detection` node in part A could be found in `rtk_challenge/lane_detection.py` in the package directory.

## Part B

In one terminal, run the following for object detection
```
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2
```

In a separate terminal, run the following which collects the coordinates of the objects detected and map them to the odom/map frames. 
```
ros2 run rtk_challenge object_mapping
```

Finally, for localisation, run
```
ros2 launch zed_wrapper rtabmap.launch.py \
localization:=true \  
rgb_topic:=/zed/zed_node/rgb/image_rect_color \
depth_topic:=/zed/zed_node/depth/depth_registered \
camera_info_topic:=/zed/zed_node/rgb/camera_info \
odom_topic:=/zed/zed_node/odom  \
visual_odometry:=false \
frame_id:=zed_camera_link \
approx_sync:=true \
wait_imu_to_init:=true \
rgbd_sync:=true \
approx_rgbd_sync:=true \
imu_topic:=/zed/zed_node/imu/data \
qos:=0 \
rviz:=true \
rtabmapviz:=true \
database_path:="<path to your map  db file>" \
initial_pose:="0 0 0 0 0 0"
```

Object coordinates and labels for part B will be saved as `object_mapping.csv` in the directory where the node is ran.
