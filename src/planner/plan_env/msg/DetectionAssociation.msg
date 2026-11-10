# One row per ORIGINAL input mask, including filtered/unmapped masks.
uint64 episode_token
uint32 observation_id
int32[] candidate_ids  # map cluster ID + 1; zero means no association
string[] reasons
float64[] bounds  # xmin,ymin,xmax,ymax for each input mask; zeros if unmapped
