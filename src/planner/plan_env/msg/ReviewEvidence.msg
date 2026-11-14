# Latest usable, unused evidence revisions. Empty list clears readiness.
uint64 episode_token
uint32 observation_id
int32[] candidate_ids
uint32[] revisions
# Current evidence assessment for the separately enabled STOP ablation.
int32[] reliable_candidate_ids
int32[] conflicting_candidate_ids
# Failed first reviews eligible for one conditional observation, fused profile only.
int32[] inspection_candidate_ids
