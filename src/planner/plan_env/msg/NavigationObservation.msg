# One simulated observation. No task goals or success measures enter this packet.
uint64 episode_token
uint32 observation_id
float64 itm_score
float64 confidence_threshold
sensor_msgs/Image depth
nav_msgs/Odometry camera_pose
nav_msgs/Odometry robot_pose
plan_env/MultipleMasksWithConfidence detections
