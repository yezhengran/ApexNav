# MultipleMasksWithConfidence.msg


# Contains multiple masks; each mask is a boolean array
sensor_msgs/PointCloud2[] point_clouds

# Corresponding confidence scores
float32[] confidence_scores

# List of label indices
int32[] label_indices

# Optional evidence provenance. Zero keeps the original mapping-only protocol.
uint64 episode_token
uint32 observation_id
std_msgs/Header header
