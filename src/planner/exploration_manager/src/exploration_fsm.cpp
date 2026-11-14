
#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>
#include <plan_env/review_rank.h>

using namespace std::chrono_literals;

namespace apexnav_planner {
void ExplorationFSM::init(rclcpp::Node::SharedPtr node)
{
  bool first_init = (node_ == nullptr);
  node_ = node;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);
  active_view_ = ActiveView();
  if (first_init) {
    node_->declare_parameter("navigation.step_sync", false);
    node_->declare_parameter("review.enabled", false);
    node_->declare_parameter("review.late_step", 350);
    node_->declare_parameter("review.lineage", false);
    node_->declare_parameter("review.stop_check", false);
    node_->declare_parameter("review.preserve_stop", false);
    node_->declare_parameter("review.automatic", false);
    node_->declare_parameter("review.fused", false);
    node_->declare_parameter("review.confidence", false);
  }
  node_->get_parameter("review.enabled", review_enabled_);
  node_->get_parameter("navigation.step_sync", step_sync_);
  observation_sequence_.reset();
  node_->get_parameter("review.late_step", review_late_step_);
  node_->get_parameter("review.lineage", review_lineage_);
  node_->get_parameter("review.stop_check", review_stop_);
  node_->get_parameter("review.preserve_stop", review_preserve_stop_);
  node_->get_parameter("review.automatic", review_automatic_);
  node_->get_parameter("review.fused", review_fused_);
  node_->get_parameter("review.confidence", review_confidence_);
  if (review_confidence_ && (!review_fused_ || review_late_step_ != 400))
    throw std::invalid_argument("Confidence profile requires fused review and step 400");
  if (review_fused_ && !review_automatic_)
    throw std::invalid_argument("Fused review requires automatic scheduling");
  if (review_automatic_ && (!review_enabled_ || !review_lineage_ || !step_sync_ || review_stop_))
    throw std::invalid_argument("Automatic review requires linked step-sync mode without STOP review");
  if (review_preserve_stop_ && !review_stop_)
    throw std::invalid_argument("Preserving STOP eligibility requires STOP review");
  if (review_stop_ && (!review_enabled_ || !review_lineage_))
    throw std::invalid_argument("STOP review requires linked selected-target review");
  if (review_late_step_ < 1) throw std::invalid_argument("review.late_step must be positive");
  review_step_ = 0;
  review_replan_ = false;
  review_pending_id_ = -1;
  review_pending_selection_ = -1;
  review_pending_high_confidence_stop_ = false;
  review_pending_automatic_ = false;
  review_pending_inspection_ = false;
  inspection_candidates_.clear();
  inspection_suppressed_at_ = -1;
  ++review_episode_;
  review_resume_path_.clear();
  review_fallback_logged_.clear();
  evidence_revisions_.clear();
  reviewed_revisions_.clear();
  reliable_evidence_.clear();
  conflicting_evidence_.clear();
  stop_reviewed_.clear();
  evidence_observation_ = 0;
  selected_mode_ = -1;

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(node_);
  expl_manager_->object_map2d_->stop_review_gating_ = review_stop_;
  expl_manager_->object_map2d_->fused_review_gating_ = review_fused_;
  expl_manager_->object_map2d_->confidence_review_ = review_confidence_;
  visualization_.reset(new PlanningVisualization(node_));
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstants::VIS_SCALE_FACTOR;

  state_ = ROS_STATE::INIT;

  if (!first_init) return;  // Skip timer/sub/pub creation on re-init (episode reset)

  observation_service_ = node_->create_service<plan_env::srv::ApplyObservation>(
      "/habitat/apply_observation",
      [this](const plan_env::srv::ApplyObservation::Request::SharedPtr request,
             plan_env::srv::ApplyObservation::Response::SharedPtr response) {
        const auto& frame = request->observation;
        response->episode_token = frame.episode_token;
        response->observation_id = frame.observation_id;
        const bool initial = frame.observation_id == 0;
        if (!step_sync_ || !observation_sequence_.accepts(frame.episode_token, frame.observation_id) ||
            (initial ? state_ != ROS_STATE::INIT :
             (state_ != ROS_STATE::WAIT_ACTION_FINISH && state_ != ROS_STATE::FINISH))) {
          response->reason = "disabled, stale/out-of-order observation, or invalid planner state";
          return;
        }
        const auto& detection = frame.detections;
        if (frame.depth.data.empty() || frame.depth.header.stamp != frame.camera_pose.header.stamp ||
            frame.depth.header.stamp != frame.robot_pose.header.stamp ||
            !std::isfinite(frame.itm_score) || !std::isfinite(frame.confidence_threshold) ||
            detection.point_clouds.size() != detection.confidence_scores.size() ||
            detection.point_clouds.size() != detection.label_indices.size() ||
            (detection.episode_token && (detection.episode_token != frame.episode_token ||
             detection.observation_id != frame.observation_id))) {
          response->reason = "inconsistent observation packet";
          return;
        }
        odometryCallback(std::make_shared<nav_msgs::msg::Odometry>(frame.robot_pose));
        auto threshold = std::make_shared<std_msgs::msg::Float64>();
        threshold->data = frame.confidence_threshold;
        confidenceThresholdCallback(threshold);
        expl_manager_->sdf_map_->applyObservation(frame);
        observation_sequence_.commit(frame.episode_token, frame.observation_id);
        expl_manager_->object_map2d_->logCandidateScores(frame.observation_id);
        response->accepted = true;
        response->reason = "applied";
        RCLCPP_INFO(node_->get_logger(),
            "[Observation applied] step=%u position=(%.6f,%.6f) itm=%.6f masks=%zu",
            frame.observation_id, fd_->odom_pos_(0), fd_->odom_pos_(1),
            frame.itm_score, detection.point_clouds.size());
      });

  evidence_sub_ = node_->create_subscription<plan_env::msg::ReviewEvidence>(
      "/habitat/review_evidence", 10,
      [this](const plan_env::msg::ReviewEvidence::SharedPtr msg) {
        if (!review_lineage_ || !msg->episode_token ||
            msg->episode_token != expl_manager_->object_map2d_->evidence_episode_token_ ||
            msg->candidate_ids.size() != msg->revisions.size() ||
            msg->observation_id < evidence_observation_) return;
        evidence_observation_ = msg->observation_id;
        reliable_evidence_.clear();
        conflicting_evidence_.clear();
        for (int id : msg->reliable_candidate_ids) if (id > 0) reliable_evidence_.insert(id - 1);
        for (int id : msg->conflicting_candidate_ids) if (id > 0) conflicting_evidence_.insert(id - 1);
        inspection_candidates_.clear();
        if (review_fused_)
          for (int id : msg->inspection_candidate_ids) if (id > 0) inspection_candidates_.insert(id - 1);
        evidence_revisions_.clear();
        for (size_t i = 0; i < msg->candidate_ids.size(); ++i)
          if (msg->candidate_ids[i] > 0 && msg->revisions[i] <= msg->observation_id)
            evidence_revisions_[msg->candidate_ids[i] - 1] = msg->revisions[i];
      });

  /* ROS2 Timer */
  exec_timer_ = node_->create_wall_timer(
      std::chrono::duration<double>(FSMConstants::EXEC_TIMER_DURATION),
      std::bind(&ExplorationFSM::FSMCallback, this));
  frontier_timer_ = node_->create_wall_timer(
      std::chrono::duration<double>(FSMConstants::FRONTIER_TIMER_DURATION),
      std::bind(&ExplorationFSM::frontierCallback, this));

  /* ROS2 Subscriber */
  trigger_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/move_base_simple/goal", 10,
      std::bind(&ExplorationFSM::triggerCallback, this, std::placeholders::_1));
  if (!step_sync_) odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
      "/odom_world", 10,
      std::bind(&ExplorationFSM::odometryCallback, this, std::placeholders::_1));
  habitat_state_sub_ = node_->create_subscription<std_msgs::msg::Int32>(
      "/habitat/state", 10,
      std::bind(&ExplorationFSM::habitatStateCallback, this, std::placeholders::_1));
  if (!step_sync_) confidence_threshold_sub_ = node_->create_subscription<std_msgs::msg::Float64>(
      "/detector/confidence_threshold", 10,
      std::bind(&ExplorationFSM::confidenceThresholdCallback, this, std::placeholders::_1));

  inspection_ack_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/habitat/inspection_ack", 10);
  inspection_sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/habitat/inspection_request", 10,
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (!review_enabled_ || review_pending_id_ < 0 ||
            (!msg->data.empty() && msg->data[0] != review_pending_id_ + 1) ||
            !active_view_.set(msg->data)) {
          RCLCPP_ERROR(node_->get_logger(), "Invalid selected-target inspection request");
          return;
        }
        std_msgs::msg::Int32 ack;
        ack.data = active_view_.id;
        inspection_ack_pub_->publish(ack);
      });
  review_target_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/habitat/review_target", 10);
  review_ack_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/habitat/review_ack", 10);
  review_result_sub_ = node_->create_subscription<std_msgs::msg::Int32MultiArray>(
      "/habitat/review_result", 10,
      [this](const std_msgs::msg::Int32MultiArray::SharedPtr msg) {
        if (!review_enabled_ || msg->data.size() != 2 || review_pending_id_ < 0 ||
            msg->data[0] != review_pending_id_ + 1) {
          RCLCPP_ERROR(node_->get_logger(), "Invalid selected-target review result");
          return;
        }
        if (review_confidence_) return;  // No unscored result may bypass the atomic channel.
        acceptReviewResult(msg->data[0], msg->data[1]);
      });

  scored_review_sub_ = node_->create_subscription<plan_env::msg::ScoredReviewResult>(
      "/habitat/scored_review_result", 10,
      [this](const plan_env::msg::ScoredReviewResult::SharedPtr msg) {
        auto map = expl_manager_->object_map2d_;
        if (!validReviewReceipt(review_confidence_, review_pending_id_, msg->candidate_id,
            msg->episode_token, map->evidence_episode_token_, msg->observation_id,
            review_step_, msg->decision, msg->target_confidence)) {
          RCLCPP_ERROR(node_->get_logger(), "Invalid atomic scored review result");
          return;
        }
        // Same callback: the numeric answer is visible before replan/ack.
        map->qwen_confidences_[review_pending_id_] = msg->target_confidence;
        acceptReviewResult(msg->candidate_id, msg->decision);
        map->logCandidateScores(msg->observation_id);
      });

  /* ROS2 Publisher */
  ros_state_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/state", 10);
  expl_state_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/expl_state", 10);
  action_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/habitat/plan_action", 10);
  expl_result_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/expl_result", 10);
  robot_marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/robot", 10);
}

// FSM between ROS and Habitat for action planning and execution
void ExplorationFSM::acceptReviewResult(int candidate, int decision) {
  const bool inconclusive = decision == ObjectMap2D::UNCERTAIN ||
      decision == ObjectMap2D::PENDING_EVIDENCE;
  const bool preserve_stop = review_preserve_stop_ &&
      review_pending_high_confidence_stop_ && inconclusive;
  if (!expl_manager_->object_map2d_->setReviewState(
          review_pending_id_, decision, preserve_stop)) {
    RCLCPP_ERROR(node_->get_logger(), "Invalid selected-target review state");
    return;
  }
  RCLCPP_INFO(node_->get_logger(), "[Review decision] episode=%d id=%d state=%d step=%d",
      review_episode_, candidate, decision, review_step_);
  active_view_ = ActiveView();
  if (review_pending_automatic_) {
    // REVIEW_TARGET is not a simulator motion. Preserve the last real
    // action for the original stuck/arrival logic. A semantic decision
    // about any candidate must never make the current route a STOP target.
    if (!review_pending_inspection_) fd_->newest_action_ = review_resume_action_;
    if (!inconclusive || review_fused_) {
      fd_->final_result_ = FINAL_RESULT::EXPLORE;
      fd_->replan_flag_ = true;
      review_replan_ = true;
    }
    std_msgs::msg::Int32 ack;
    ack.data = candidate;
    review_pending_id_ = -1;
    review_pending_automatic_ = false;
    review_pending_inspection_ = false;
    review_ack_pub_->publish(ack);
    return;
  }
  expl_manager_->ed_->next_pos_ = review_resume_pos_;
  expl_manager_->ed_->next_best_path_ = review_resume_path_;
  const bool fallback = review_step_ >= review_late_step_ ||
      review_pending_selection_ == EXPL_RESULT::SEARCH_SUSPICIOUS_OBJECT ||
      review_pending_selection_ == EXPL_RESULT::SEARCH_EXTREME;
  if (preserve_stop)
    RCLCPP_INFO(node_->get_logger(), "[Review preserve STOP] episode=%d id=%d state=%d step=%d",
        review_episode_, candidate, decision, review_step_);
  if (inconclusive && fallback &&
      review_fallback_logged_.insert(review_pending_id_).second)
    RCLCPP_INFO(node_->get_logger(), "[Review fallback] episode=%d id=%d reason=%s step=%d",
        review_episode_, candidate,
        review_step_ >= review_late_step_ ? "step_budget" : "no_passable_frontier", review_step_);
  fd_->final_result_ = (decision == ObjectMap2D::CONFIRMED ||
      (inconclusive && (fallback || preserve_stop))) ?
      FINAL_RESULT::SEARCH_OBJECT : FINAL_RESULT::EXPLORE;
  fd_->replan_flag_ = true;
  review_replan_ = true;  // A revoked target must not reuse its old route.
  std_msgs::msg::Int32 ack;
  ack.data = candidate;
  review_pending_id_ = -1;
  review_pending_high_confidence_stop_ = false;
  // SEARCH_OBJECT returns through the original arrival checks; never issue STOP here.
  review_ack_pub_->publish(ack);
}

void ExplorationFSM::FSMCallback()
{
  exec_timer_->cancel();
  std_msgs::msg::Int32 ros_state_msg;
  ros_state_msg.data = state_;
  ros_state_pub_->publish(ros_state_msg);
  switch (state_) {
    case ROS_STATE::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
            "No odom || No target confidence threshold.");
        exec_timer_->reset();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(ROS_STATE::WAIT_TRIGGER, "FSM");
      break;
    }

    case ROS_STATE::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Wait for trigger.");
      break;
    }

    case ROS_STATE::FINISH: {
      if (!fd_->have_finished_) {
        fd_->have_finished_ = true;
        clearVisMarker();
        std_msgs::msg::Int32 action_msg;
        action_msg.data = ACTION::STOP;
        action_pub_->publish(action_msg);
      }
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Finish One Episode!!!");
      break;
    }

    case ROS_STATE::PLAN_ACTION: {
      // Review qualified cached frames even during the initial scan. This does
      // not consume a motion, increment the scan counter, or change arrival.
      if (maybeReviewAutomatic()) {
        transitState(ROS_STATE::PUB_ACTION, "automatic_review");
        break;
      }
      // Initial action sequence: perform orientation calibration turns
      if (fd_->init_action_count_ < 1 + 12 + 1 + 12 && !(review_fused_ && active_view_.id)) {
        if (fd_->init_action_count_ < 1)
          fd_->newest_action_ = ACTION::TURN_DOWN;
        else if (fd_->init_action_count_ < 1 + 12)
          fd_->newest_action_ = ACTION::TURN_LEFT;
        else if (fd_->init_action_count_ < 1 + 12 + 1)
          fd_->newest_action_ = ACTION::TURN_UP;
        else
          fd_->newest_action_ = ACTION::TURN_LEFT;
        RCLCPP_WARN(node_->get_logger(), "Init Mode Process -----> (%d/26)", fd_->init_action_count_);
        fd_->init_action_count_++;
        transitState(ROS_STATE::PUB_ACTION, "FSM");
        updateFrontierAndObject();
      }
      else {
        // Main planning phase: determine robot pose and call action planner
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;

        auto t1 = node_->get_clock()->now();
        fd_->final_result_ = callActionPlanner();
        double call_action_planner_time = (node_->get_clock()->now() - t1).seconds();
        RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 10000,
            "[Calculating Time] Planning process time = %.3f s", call_action_planner_time);

        std_msgs::msg::Int32 expl_state_msg;
        expl_state_msg.data = fd_->final_result_;
        expl_state_pub_->publish(expl_state_msg);
        if (fd_->final_result_ == FINAL_RESULT::EXPLORE ||
            fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT)
          transitState(ROS_STATE::PUB_ACTION, "FSM");
        else
          transitState(ROS_STATE::FINISH, "FSM");
      }
      visualize();
      break;
    }

    case ROS_STATE::PUB_ACTION: {
      std_msgs::msg::Int32 action_msg;
      action_msg.data = fd_->newest_action_;
      action_pub_->publish(action_msg);
      transitState(ROS_STATE::WAIT_ACTION_FINISH, "FSM");
      break;
    }

    case ROS_STATE::WAIT_ACTION_FINISH: {
      exec_timer_->reset();
      break;
    }
  }
  exec_timer_->reset();
}

/**
 * @brief Plan the next action based on current state and environment
 * @return Final result indicating the planned action type and exploration state
 *
 * This is the core planning function that decides what action the robot should take next.
 * It handles obstacle avoidance, frontier exploration, object search, and stuck recovery.
 */
int ExplorationFSM::callActionPlanner()
{
  const double stucking_distance = FSMConstants::STUCKING_DISTANCE;
  const double reach_distance = FSMConstants::REACH_DISTANCE;
  const double soft_reach_distance = FSMConstants::SOFT_REACH_DISTANCE;

  bool frontier_change_flag = updateFrontierAndObject();

  int expl_res, final_res;
  Eigen::Vector2d current_pos = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));
  Eigen::Vector2d last_pos = Eigen::Vector2d(fd_->last_start_pos_(0), fd_->last_start_pos_(1));
  double current_yaw = fd_->start_yaw_(0);
  fd_->last_start_pos_ = fd_->start_pt_;

  // Inspection is a bounded detour, never a REACH_OBJECT/STOP condition.
  // With no request the original planner takes exactly the baseline path.
  if (active_view_.id) return planInspection();

  // Reach the object - check if close enough to target object
  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (current_pos - expl_manager_->ed_->next_pos_).norm() < reach_distance) {
    if (maybeReviewStop()) return FINAL_RESULT::SEARCH_OBJECT;
    RCLCPP_ERROR(node_->get_logger(), "Reach the object successfully!!!");
    final_res = FINAL_RESULT::REACH_OBJECT;
    return final_res;
  }

  /*******  Escape-from-stuck logic START *******/
  // Detect if robot is stuck and initiate escape sequence
  int last_action = fd_->newest_action_;
  if (!fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() < stucking_distance &&
      last_action == ACTION::MOVE_FORWARD) {
    if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
        (current_pos - expl_manager_->ed_->next_pos_).norm() < soft_reach_distance) {
      if (maybeReviewStop()) return FINAL_RESULT::SEARCH_OBJECT;
      RCLCPP_ERROR(node_->get_logger(), "Reach the object successfully!!!");
      final_res = FINAL_RESULT::REACH_OBJECT;
      return final_res;
    }

    bool past_stucking_flag = false;
    for (auto stucking_point : fd_->stucking_points_) {
      Vector2d stucking_pos = Vector2d(stucking_point(0), stucking_point(1));
      double stucking_yaw = stucking_point(2);
      if ((stucking_pos - current_pos).norm() < stucking_distance &&
          fabs(stucking_yaw - current_yaw) < FSMConstants::ACTION_ANGLE) {
        past_stucking_flag = true;
        RCLCPP_ERROR(node_->get_logger(), "Still stuck at the same place");
        break;
      }
    }
    if (!past_stucking_flag) {
      fd_->escape_stucking_flag_ = true;
      fd_->escape_stucking_count_ = 0;
      fd_->escape_stucking_pos_ = current_pos;
      fd_->escape_stucking_yaw_ = current_yaw;
    }
  }

  if (fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() >= stucking_distance) {
    RCLCPP_ERROR(node_->get_logger(), "Escaped from stuck state.");
    fd_->escape_stucking_flag_ = false;
  }

  if (fd_->escape_stucking_flag_) {
    RCLCPP_ERROR(node_->get_logger(), "Escaping stuck...");
    if (fd_->escape_stucking_count_ == 0)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 1)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 2)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 3)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 4)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 5)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 6)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 7)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 8)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 9)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else {
      // Failed to escape - mark area as occupied and add to stuck points
      RCLCPP_ERROR(node_->get_logger(), "Cannot escape stuck state.");
      fd_->escape_stucking_flag_ = false;
      expl_manager_->sdf_map_->setForceOccGrid(current_pos);
      double forward_distance = FSMConstants::FORWARD_DISTANCE;
      Eigen::Vector2d forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      forward_distance = FSMConstants::FORWARD_DISTANCE * 2.0;
      forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      fd_->dormant_frontier_flag_ = true;
      Vector3d stucking_point(
          fd_->escape_stucking_pos_(0), fd_->escape_stucking_pos_(1), fd_->escape_stucking_yaw_);
      fd_->stucking_points_.push_back(stucking_point);
    }

    if (fd_->escape_stucking_flag_) {
      fd_->escape_stucking_count_++;
      return fd_->final_result_;
    }
  }

  /*******  Decide whether to replan path (stability heuristic) START *******/
  // Use path stability to reduce oscillation between different frontier targets
  vector<Vector2d> last_next_best_path = expl_manager_->ed_->next_best_path_;
  Vector2d last_next_pos = expl_manager_->ed_->next_pos_;
  if (fd_->dormant_frontier_flag_) {
    fd_->replan_flag_ = true;
    fd_->dormant_frontier_flag_ = false;
  }
  else if (fd_->final_result_ == FINAL_RESULT::EXPLORE && !frontier_change_flag && !review_replan_)
    fd_->replan_flag_ = false;

  expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));

  if (expl_res != EXPL_RESULT::EXPLORATION) {
    fd_->replan_flag_ = true;
  }
  if (expl_res == EXPL_RESULT::EXPLORATION && !fd_->replan_flag_) {
    expl_manager_->ed_->next_best_path_ = last_next_best_path;
    expl_manager_->ed_->next_pos_ = last_next_pos;
    fd_->replan_flag_ = true;
  }
  review_replan_ = false;
  selected_mode_ = expl_res;
  /*******  Decide whether to replan path (stability heuristic) END *******/

  // Publish exploration result to monitor
  std_msgs::msg::Int32 expl_result_msg;
  expl_result_msg.data = expl_res;
  expl_result_pub_->publish(expl_result_msg);

  // Determine current high-level state based on exploration results
  if (expl_res == EXPL_RESULT::EXPLORATION)
    final_res = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    final_res = FINAL_RESULT::NO_FRONTIER;
  else
    final_res = FINAL_RESULT::SEARCH_OBJECT;

  if (final_res == FINAL_RESULT::NO_FRONTIER || expl_manager_->ed_->next_best_path_.empty()) {
    RCLCPP_WARN(node_->get_logger(), "No (passable) frontier");
    return final_res;
  }

  if (review_enabled_ && maybeReviewSelected(expl_res))
    return FINAL_RESULT::SEARCH_OBJECT;

  Eigen::Vector2d end_pos = expl_manager_->ed_->next_pos_;
  Eigen::Vector2d last_end_pos = fd_->last_next_pos_;
  fd_->last_next_pos_ = end_pos;
  double min_dist = (current_pos - end_pos).norm();
  RCLCPP_WARN(node_->get_logger(), "To the next point (%.2fm %.2fm), distance = %.2f m",
      end_pos(0), end_pos(1), min_dist);

  // Handling being stuck while exploring toward a specific frontier
  if (final_res == FINAL_RESULT::EXPLORE) {
    // Force dormant if very close to target but still exploring
    if (min_dist < FSMConstants::FORCE_DORMANT_DISTANCE) {
      RCLCPP_ERROR(node_->get_logger(), "Force set dormant frontier.");
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }

    // Count consecutive times with same target position while stuck
    if ((end_pos - last_end_pos).norm() < 1e-3 &&
        (current_pos - last_pos).norm() < stucking_distance) {
      fd_->stucking_next_pos_count_++;
      if (fd_->stucking_next_pos_count_ > 8) {
        RCLCPP_ERROR(node_->get_logger(), "stucking_next_pos_count_ = %d",
            fd_->stucking_next_pos_count_);
      }
    }
    else
      fd_->stucking_next_pos_count_ = 0;

    // Mark frontier as dormant if stuck too long with same target
    if (fd_->stucking_next_pos_count_ >= FSMConstants::MAX_STUCKING_NEXT_POS_COUNT) {
      RCLCPP_ERROR(node_->get_logger(), "Set dormant frontier.");
      fd_->stucking_action_count_ = 0;
      fd_->stucking_next_pos_count_ = 0;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }
  }

  // Track consecutive stuck actions globally
  if ((current_pos - last_pos).norm() < stucking_distance) {
    fd_->stucking_action_count_++;
    if (fd_->stucking_action_count_ > 15) {
      RCLCPP_ERROR(node_->get_logger(), "Stucking action count = %d",
          fd_->stucking_action_count_);
    }
  }
  else
    fd_->stucking_action_count_ = 0;

  // If stuck for too long globally, terminate episode
  if (fd_->stucking_action_count_ >= FSMConstants::MAX_STUCKING_COUNT) {
    RCLCPP_ERROR(node_->get_logger(), "Stuck for too long, stopping episode.");
    final_res = FINAL_RESULT::STUCKING;
    return final_res;
  }

  // Plan specific action based on exploration result
  if (expl_res == EXPL_RESULT::SEARCH_EXTREME)
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_, false);
  else
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_);

  return final_res;
}

bool ExplorationFSM::maybeReviewSelected(int exploration_result)
{
  if (review_automatic_) return false;
  const int id = expl_manager_->ed_->selected_object_id_;
  vector<double> data;
  auto& objects = expl_manager_->object_map2d_;
  if (!objects->reviewInfo(id, data)) {
    if (exploration_result != EXPL_RESULT::EXPLORATION)
      RCLCPP_INFO(node_->get_logger(),
          "[Review bypass] episode=%d reason=missing_object_metadata selection=%d step=%d",
          review_episode_, exploration_result, review_step_);
    return false;
  }
  // High-confidence selection retains the complete original path.
  if (data[1] >= data[2]) return false;
  const int status = objects->reviewState(id);
  const bool inconclusive = status == ObjectMap2D::UNCERTAIN ||
      status == ObjectMap2D::PENDING_EVIDENCE;
  if (inconclusive && review_fallback_logged_.insert(id).second) {
    RCLCPP_INFO(node_->get_logger(), "[Review fallback] episode=%d id=%d reason=%s step=%d",
        review_episode_, id + 1,
        review_step_ >= review_late_step_ ? "step_budget" : "no_passable_frontier", review_step_);
  }
  if (status != ObjectMap2D::UNREVIEWED &&
      !(review_lineage_ && inconclusive && evidence_revisions_[id] > reviewed_revisions_[id]))
    return false;
  reviewed_revisions_[id] = evidence_revisions_[id];
  return requestReview(id, exploration_result, data, false);
}

bool ExplorationFSM::maybeReviewStop()
{
  if (!review_stop_) return false;
  const int id = expl_manager_->ed_->selected_object_id_;
  vector<double> info;
  if (!expl_manager_->object_map2d_->reviewInfo(id, info)) {
    RCLCPP_INFO(node_->get_logger(), "[Stop review bypass] reason=missing_object_metadata step=%d", review_step_);
    return false;
  }
  // One bounded STOP review per candidate; inconclusive results retain the
  // original late/no-frontier fallback. Do not create a zero-step retry loop.
  if (stop_reviewed_.count(id)) return false;
  const bool fresh = evidence_observation_ > 0 &&
      evidence_observation_ + 2 >= static_cast<uint32_t>(review_step_);
  const bool conflict = fresh && conflicting_evidence_.count(id);
  if (fresh && reliable_evidence_.count(id) && !conflict) {
    RCLCPP_INFO(node_->get_logger(), "[Stop review bypass] id=%d reason=current_linked_evidence step=%d", id + 1, review_step_);
    return false;
  }
  stop_reviewed_.insert(id);
  RCLCPP_INFO(node_->get_logger(), "[Stop review trigger] id=%d reason=%s step=%d", id + 1,
      conflict ? "category_conflict" : "missing_current_evidence", review_step_);
  return requestReview(id, selected_mode_, info, true);
}

bool ExplorationFSM::maybeReviewAutomatic()
{
  if (!review_automatic_ || review_pending_id_ >= 0 || active_view_.id) return false;
  auto& objects = expl_manager_->object_map2d_;
  // Python publishes only threshold-qualified, novel, budget-eligible frames.
  // Sort IDs so callback/unordered-map iteration cannot affect scheduling.
  std::set<int> candidates;
  for (const auto& entry : evidence_revisions_) candidates.insert(entry.first);
  for (int id : candidates) {
    const int status = objects->reviewState(id);
    if (status == ObjectMap2D::CONFIRMED || status == ObjectMap2D::REJECTED) continue;
    if (reviewed_revisions_.count(id) && evidence_revisions_.at(id) <= reviewed_revisions_.at(id)) continue;
    vector<double> data;
    if (!objects->reviewInfo(id, data)) continue;
    if (review_fused_ && !objects->reviewEligible(id)) continue;
    reviewed_revisions_[id] = evidence_revisions_.at(id);
    RCLCPP_INFO(node_->get_logger(), "[Automatic review trigger] id=%d evidence_step=%u step=%d",
        id + 1, evidence_revisions_.at(id), review_step_);
    return requestReview(id, selected_mode_, data, false, true);
  }
  if (review_fused_ && !inspection_candidates_.empty()) {
    const int alternative = expl_manager_->reachableConfirmedTarget(fd_->odom_pos_);
    if (alternative >= 0) {
      if (inspection_suppressed_at_ != review_step_) {
        RCLCPP_INFO(node_->get_logger(), "[Inspection deferred] confirmed_id=%d pending=%zu step=%d",
            alternative + 1, inspection_candidates_.size(), review_step_);
        inspection_suppressed_at_ = review_step_;
      }
      return false;
    }
    for (int id : inspection_candidates_) {
      vector<double> data;
      if (objects->reviewState(id) != ObjectMap2D::UNCERTAIN ||
          !objects->reviewEligible(id, false) || !objects->reviewInfo(id, data)) continue;
      RCLCPP_INFO(node_->get_logger(), "[Conditional inspection trigger] id=%d step=%d", id + 1, review_step_);
      return requestReview(id, selected_mode_, data, false, true, true);
    }
  }
  return false;
}

bool ExplorationFSM::requestReview(int id, int exploration_result, const vector<double>& data,
    bool at_stop, bool automatic, bool inspection)
{
  review_pending_id_ = id;
  review_pending_selection_ = exploration_result;
  review_pending_high_confidence_stop_ = at_stop && data[1] >= data[2];
  review_pending_automatic_ = automatic;
  review_pending_inspection_ = inspection;
  review_resume_action_ = fd_->newest_action_;
  if (!automatic) {
    review_resume_pos_ = expl_manager_->ed_->next_pos_;
    review_resume_path_ = expl_manager_->ed_->next_best_path_;
  }
  std_msgs::msg::Float64MultiArray target;
  target.data = data;
  if (at_stop) target.data.push_back(1.0);
  if (automatic) target.data.push_back(review_fused_ ? (inspection ? 4.0 : 3.0) : 2.0);
  review_target_pub_->publish(target);
  fd_->newest_action_ = ACTION::REVIEW_TARGET;
  RCLCPP_INFO(node_->get_logger(),
      "[Review selected] episode=%d id=%d confidence=%.4f threshold=%.4f step=%d",
      review_episode_, id + 1, data[1], data[2], review_step_);
  return true;
}

int ExplorationFSM::planInspection()
{
  const Vector2d position(fd_->start_pt_(0), fd_->start_pt_(1));
  if (active_view_.turn_first && !active_view_.oriented) {
    const auto direction = (active_view_.center - position).eval();
    double difference = std::atan2(direction.y(), direction.x()) - fd_->start_yaw_(0);
    wrapAngle(difference);
    if (direction.norm() > 0.05 && std::abs(difference) > 18.0 * M_PI / 180.0 &&
        active_view_.orientation_steps < 6) {
      ++active_view_.orientation_steps;
      fd_->newest_action_ = difference > 0 ? ACTION::TURN_LEFT : ACTION::TURN_RIGHT;
      return FINAL_RESULT::EXPLORE;
    }
    // Each real turn already supplied a fresh frame to the Python reviewer.
    // If still uncertain, continue the existing size-adaptive safe route.
    active_view_.oriented = true;
  }
  auto map = expl_manager_->sdf_map_;
  auto clear = [map](const Vector2d& p) {
    return map->isInMap(p) && map->getOccupancy(p) != SDFMap2D::UNKNOWN &&
        map->getOccupancy(p) != SDFMap2D::OCCUPIED;
  };
  auto safe = [map, clear](const Vector2d& p) {
    return clear(p) && map->getInflateOccupancy(p) != 1;
  };
  auto route = [this, position](const Vector2d& goal, vector<Vector2d>& path) {
    auto& astar = expl_manager_->path_finder_;
    astar->reset();
    if (astar->astarSearch(position, goal, 0.10, 0.05, Astar2D::SAFETY_MODE::NORMAL) !=
        Astar2D::REACH_END) return false;
    path = astar->getPath();
    return true;
  };
  vector<Vector2d> path;
  const bool was_locked = active_view_.locked;
  if (!active_view_.plan(position, safe, clear, route, path)) {
    fd_->newest_action_ = ACTION::INSPECTION_FAILED;
    return FINAL_RESULT::EXPLORE;
  }
  if (!was_locked)
    RCLCPP_INFO(node_->get_logger(), "[Active view] id=%d point=(%.3f, %.3f) radius=%.3f",
        active_view_.id, active_view_.goal.x(), active_view_.goal.y(), active_view_.radius);
  expl_manager_->ed_->next_pos_ = active_view_.goal;
  expl_manager_->ed_->next_best_path_ = path;
  if ((position - active_view_.goal).norm() <= 0.3125) {
    if (!active_view_.visible(position, clear)) {
      fd_->newest_action_ = ACTION::INSPECTION_FAILED;
      return FINAL_RESULT::EXPLORE;
    }
    double yaw = std::atan2(active_view_.center.y() - position.y(),
                           active_view_.center.x() - position.x());
    double difference = yaw - fd_->start_yaw_(0);
    wrapAngle(difference);
    if (std::abs(difference) <= 18.0 * M_PI / 180.0)
      fd_->newest_action_ = ACTION::INSPECT;
    else
      fd_->newest_action_ = difference > 0 ? ACTION::TURN_LEFT : ACTION::TURN_RIGHT;
  } else {
    fd_->newest_action_ = planNextBestAction(position, fd_->start_yaw_(0), path);
    // Quantized execution follows the current heading, which can differ
    // from the continuous path direction by half a turn increment.
    if (fd_->newest_action_ == ACTION::MOVE_FORWARD) {
      Vector2d end = position + 0.25 * Vector2d(
          std::cos(fd_->start_yaw_(0)), std::sin(fd_->start_yaw_(0)));
      if (!ActiveView::safePath({position, end}, safe))
        fd_->newest_action_ = ACTION::INSPECTION_FAILED;
    }
  }
  return FINAL_RESULT::EXPLORE;
}

int ExplorationFSM::planNextBestAction(
    Vector2d current_pos, double current_yaw, const vector<Vector2d>& path, bool need_safety)
{
  const double local_distance = FSMConstants::LOCAL_DISTANCE;

  // Update target position based on path and local distance
  Vector2d local_pos = selectLocalTarget(current_pos, path, local_distance);
  fd_->local_pos_ = local_pos;

  // Compute the best step considering obstacles and safety
  Vector2d best_step;
  if ((current_pos - path.back()).norm() > FSMConstants::ACTION_DISTANCE && need_safety)
    best_step = computeBestStep(current_pos, current_yaw, local_pos);
  else
    best_step = local_pos;

  // Calculate target orientation from best step direction
  double target_yaw = std::atan2(best_step(1) - current_pos(1), best_step(0) - current_pos(0));
  return decideNextAction(current_yaw, target_yaw);
}

Vector2d ExplorationFSM::selectLocalTarget(
    const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance)
{
  Vector2d target_pos = path.back();

  // Find the closest path point to current position as starting search index
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select a local target position within the specified distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      break;
    }
  }

  return target_pos;
}

Vector2d ExplorationFSM::computeBestStep(
    const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos)
{
  Vector2d best_step = target_pos;

  double min_cost = std::numeric_limits<double>::max();
  for (auto step : fp_->action_steps_) {
    double cost = computeActionTotalCost(current_pos, current_yaw, target_pos, step);
    if (cost < min_cost) {
      best_step = current_pos + step;
      min_cost = cost;
    }
  }

  return best_step;
}

// Compute total cost of taking a step towards target
// Considers distance-to-target, movement efficiency, and collision safety
double ExplorationFSM::computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
    const Vector2d& target_pos, const Vector2d& step)
{
  const double traget_weight = FSMConstants::TARGET_WEIGHT;
  const double traget_close_weight1 = FSMConstants::TARGET_CLOSE_WEIGHT_1;
  const double traget_close_weight2 = FSMConstants::TARGET_CLOSE_WEIGHT_2;
  const double safety_weight = FSMConstants::SAFETY_WEIGHT;
  double cost = 0.0;

  // Distance-to-target cost
  Vector2d step_pos = current_pos + step;
  double target_cost = traget_weight * (step_pos - target_pos).norm();

  // Change-in-distance cost (negative if moving closer)
  double target_close_cost = (step_pos - target_pos).norm() - (current_pos - target_pos).norm();
  if (target_close_cost > 0)
    target_close_cost *= traget_close_weight1;
  else
    target_close_cost *= traget_close_weight2;

  // Safety distance cost
  double safety_cost = safety_weight * computeActionSafetyCost(current_pos, step);

  cost += target_cost + target_close_cost + safety_cost;
  return cost;
}

// Compute safety cost along the step using SDF distance to obstacles
// Returns higher cost for paths that go too close to obstacles
double ExplorationFSM::computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step)
{
  const double min_safe_distance = FSMConstants::MIN_SAFE_DISTANCE;
  const double sample_num = FSMConstants::SAMPLE_NUM;

  Vector2d dir = step;
  double len = dir.norm();
  dir.normalize();

  double safety_cost = 0.0;
  for (double l = len / sample_num; l < len; l += len / sample_num) {
    Vector2d ckpt = current_pos + l * dir;
    Vector2d grad;
    double dist_to_occ = expl_manager_->sdf_map_->getDistWithGrad(ckpt, grad);
    if (dist_to_occ < min_safe_distance)
      safety_cost += 1 / (dist_to_occ + 1e-2);
  }

  return safety_cost;
}

// Decide whether to turn or move forward based on yaw difference
// Uses action angle threshold to determine if orientation adjustment is needed
int ExplorationFSM::decideNextAction(double current_yaw, double target_yaw)
{
  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  int next_action;
  if (std::fabs(yaw_diff) > FSMConstants::ACTION_ANGLE / 1.9) {
    if (yaw_diff > 0)
      next_action = ACTION::TURN_LEFT;
    else
      next_action = ACTION::TURN_RIGHT;
  }
  else
    next_action = ACTION::MOVE_FORWARD;

  return next_action;
}

void ExplorationFSM::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  // Lambda function to convert 2D vectors to 3D for visualization
  auto vec2dTo3d = [](const vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontier
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontier
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw object
  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw next local point
  vector<Vector2d> local_points;
  local_points.push_back(fd_->local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);

  visualization_->drawSpheres(vec2dTo3d(fd_->traveled_path_), fp_->vis_scale_ * 1.5,
      Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), "traveled_path", 1, 6);
}

void ExplorationFSM::clearVisMarker()
{
  auto ed_ptr = expl_manager_->ed_;
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }

  visualization_->drawLines({}, fp_->vis_scale_, Vector4d(0, 0, 1, 1), "next_path", 1, 6);
}

bool ExplorationFSM::updateFrontierAndObject()
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  Eigen::Vector2d start_pos2d = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(start_pos2d, fd_->start_yaw_(0));
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

// Receive Habitat state messages
void ExplorationFSM::habitatStateCallback(const std_msgs::msg::Int32::SharedPtr msg)
{
  if (review_enabled_ && msg->data == HABITAT_STATE::ACTION_EXEC) {
    ++review_step_;  // Only real Habitat actions; review handshakes never count.
    expl_manager_->object_map2d_->setLateReviewEligibility(review_step_ >= review_late_step_);
  }
  if (msg->data == HABITAT_STATE::ACTION_FINISH && state_ == ROS_STATE::WAIT_ACTION_FINISH)
    transitState(PLAN_ACTION, "Habitat Finish Action");
  if (msg->data == HABITAT_STATE::EPISODE_FINISH)
    init(node_);
  return;
}

// Periodically update frontiers and visualize in idle states
void ExplorationFSM::frontierCallback()
{
  if (state_ != ROS_STATE::WAIT_TRIGGER && state_ != ROS_STATE::FINISH)
    return;

  updateFrontierAndObject();
  visualize();
}

// Receive user trigger to start exploration
void ExplorationFSM::triggerCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER)
    return;
  fd_->trigger_ = true;
  std::cout << "Triggered!" << std::endl;
  transitState(PLAN_ACTION, "triggerCallback");
}

// Receive robot odometry and update traveled path + marker
void ExplorationFSM::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;

  Vector2d odom_pos2d = Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));
  if (fd_->traveled_path_.empty())
    fd_->traveled_path_.push_back(odom_pos2d);
  else if ((fd_->traveled_path_.back() - odom_pos2d).norm() > 1e-2)
    fd_->traveled_path_.push_back(odom_pos2d);

  publishRobotMarker();
}

void ExplorationFSM::publishRobotMarker()
{
  const double robot_height = FSMConstants::ROBOT_HEIGHT;
  const double robot_radius = FSMConstants::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::msg::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = node_->get_clock()->now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::msg::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::msg::Marker::ADD;

  // Set cylinder position
  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  // Set cylinder orientation
  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set cylinder dimensions
  robot_marker.scale.x = robot_radius * 2;  // Diameter
  robot_marker.scale.y = robot_radius * 2;  // Diameter
  robot_marker.scale.z = robot_height;      // Height

  // Set cylinder color (blue)
  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::msg::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = node_->get_clock()->now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
  arrow_marker.action = visualization_msgs::msg::Marker::ADD;

  // Set arrow position
  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  // Set arrow orientation
  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set arrow dimensions
  arrow_marker.scale.x = robot_radius + 0.13;  // Arrow length
  arrow_marker.scale.y = 0.08;                 // Arrow width
  arrow_marker.scale.z = 0.08;                 // Arrow thickness

  // Set arrow color (green)
  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  // Publish both markers
  robot_marker_pub_->publish(robot_marker);
  robot_marker_pub_->publish(arrow_marker);
}

void ExplorationFSM::confidenceThresholdCallback(const std_msgs::msg::Float64::SharedPtr msg)
{
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
}

// Transition FSM state and log the change
void ExplorationFSM::transitState(ROS_STATE new_state, string pos_call)
{
  int pre_s = int(state_);
  state_ = new_state;
  std::cout << "[ " + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " +
              fd_->state_str_[int(new_state)]
       << std::endl;
}
}  // namespace apexnav_planner
