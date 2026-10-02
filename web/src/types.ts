export interface ObjectiveCompletion {
  kind: 'equipment' | 'inventory_item' | 'quest_item' | 'story_flag' | 'scene' | 'scene_room' | 'leave_scene_room' | 'rupees_at_least' | 'heart_pieces_at_least' | 'skull_tokens_at_least' | 'small_keys_at_least' | 'magic_acquired' | 'dialogue_actor' | 'event_kind' | 'game_completed' | 'manual';
  name: string | null;
  item_id: number | null;
  flag: string | null;
  scene: number | null;
  room: number | null;
  threshold: number | null;
  actor_id: number | null;
  actor_name: string | null;
  event_kind: string | null;
}

export interface AgentIntent {
  objective: string;
  completion: ObjectiveCompletion;
  summary: string;
  mode: 'explore' | 'navigate' | 'interact' | 'combat' | 'dialogue' | 'menu' | 'observe';
  target_actor_id: number | null;
  target_actor_params: number | null;
  target_actor_uid: string | null;
  target_position: number[] | null;
  target_item_id: number | null;
  direction: 'forward' | 'back' | 'left' | 'right' | 'up' | 'down' | null;
  choice_index: number | null;
  horizon_ms: number;
}

export interface InputState {
  buttons: number;
  button_names: string[];
  stick_x: number;
  stick_y: number;
  reason: string;
}

export interface QuotaWindow {
  limit_id: string | null;
  limit_name: string | null;
  slot: string;
  used_percent: number;
  remaining_percent: number;
  window_duration_mins: number | null;
  resets_at: number | null;
}

export interface ProviderQuota {
  available: boolean;
  connected?: boolean;
  plan?: string | null;
  plan_type?: string | null;
  ordinary_usage_allowed?: boolean | null;
  windows: QuotaWindow[];
  credits?: {
    has_credits: boolean;
    unlimited: boolean;
    balance: string | number | null;
  } | null;
  individual_limit?: {
    limit: string | number | null;
    used: string | number | null;
    remaining_percent: number | null;
    resets_at: number | null;
  } | null;
  reset_credits?: { available_count: number | null } | null;
  rate_limit_reached_type?: string | null;
  limit?: number | null;
  limit_remaining?: number | null;
  error?: string | null;
}

export interface RunUsageBreakdown {
  provider: string;
  model: string;
  effort: string | null;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  reasoning_output_tokens: number;
  total_tokens: number;
  unknown_usage_calls: number;
  unknown_cost_calls: number;
  known_cost_usd: number;
  cost_usd: number | null;
}

export interface UsageSnapshot {
  provider: string | null;
  model: string | null;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  reasoning_output_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
  known_cost_usd: number;
  quota: ProviderQuota;
  quota_updated_at: number | null;
}


export interface LearningAchievement {
  id: string;
  kind: string;
  title: string;
  detail: string;
  points: number;
  training_reward: number;
  score_after: number;
  action_index: number;
}

export interface LearningSnapshot {
  learner_device?: string;
  actor_device?: string;
  updates?: number;
  samples_trained?: number;
  run_updates?: number;
  run_samples_trained?: number;
  rollout_steps?: number;
  queued_rollouts?: number;
  last_reward?: number;
  total_reward?: number;
  recent_mean_reward?: number;
  positive_reward_rate?: number;
  useful_progress_rate?: number;
  objective_score?: number;
  button_probability_mean?: number;
  expected_button_count?: number;
  guidance_mix?: number;
  interaction_learning?: {
    learned: number;
    probe_successes: number;
    pending: boolean;
    last: string;
    dialogue_reentry_guard?: boolean;
    dialogue_reentry_suppressed?: number;
  };
  stick_entropy?: number;
  button_entropy?: number;
  stick_entropy_coef?: number;
  button_entropy_coef?: number;
  current_stick_entropy_coef?: number;
  current_button_entropy_coef?: number;
  exploration_decay?: number;
  achievements?: LearningAchievement[];
  resources?: {
    chests_opened: number;
    rupees_collected: number;
    ammo_collected: number;
    health_recovered: number;
    magic_recovered: number;
  };
  exploration?: {
    unique_spaces: number;
    unique_actors: number;
    unique_transitions: number;
    unique_macro_regions?: number;
    local_dwell_seconds?: number;
    local_anchor_distance?: number;
    local_frontier_radius?: number;
    local_dwell_penalty?: number;
  };
  reward_breakdown?: Record<string, number>;
  room_map?: {
    rooms: number;
    entries: number;
    transitions: number;
    exits: number;
    doors: number;
    walkable_cells: number;
    blocked_cells: number;
    affordances: number;
    writable: boolean;
    dirty?: boolean;
    load_error?: string | null;
    save_error?: string | null;
    current_room?: {
      scene: number;
      room: number;
      visits: number;
      entries: number;
      transitions: number;
      exits: number;
      doors: number;
      walkable_cells: number;
      blocked_cells: number;
      affordances: number;
    } | null;
  };
  route_memory?: {
    nodes: number;
    edges: number;
    learned_interactions?: number;
    routes_reused: number;
    waypoints_advanced?: number;
    active_frontier?: string | null;
    frontier_completed?: number;
    frontier_abandoned?: number;
    frontier_failed_cells?: number;
    frontier_failure_total?: number;
    room_failure_pressure?: number;
    route_edge_completed?: number;
    route_edge_abandoned?: number;
    route_edges_cooling_down?: number;
    active_route_edge?: string | null;
    last_path_nodes: number;
    cached_path_nodes?: number;
    exhausted_partial_nodes?: number;
    revision?: number;
    persistence_revision?: number;
    dirty?: boolean;
    last_target_gap: number | null;
    writable: boolean;
    load_error?: string | null;
    save_error?: string | null;
  };
  last_update?: Record<string, unknown>;
  checkpoint_load_error?: string;
  training_enabled?: boolean;
  rnd_error_ema?: number | null;
  rnd_last_error?: number;
  rnd_last_novelty?: number;
}


export interface ChampionSummary {
  id: string;
  kind: string;
  created_at: number;
  run_id: string;
  elapsed_s: number;
  contract?: string;
  updates?: number;
  samples_trained?: number;
  run_updates?: number;
  run_samples_trained?: number;
  objective_score?: number;
  total_reward?: number;
  provider?: string | null;
  model?: string | null;
  effort?: string | null;
  calls?: number;
  input_tokens?: number;
  output_tokens?: number;
  cached_input_tokens?: number;
  reasoning_output_tokens?: number;
  total_tokens?: number;
  cost_usd?: number | null;
  known_cost_usd?: number;
  unknown_usage_calls?: number;
  unknown_cost_calls?: number;
  usage_by_model?: RunUsageBreakdown[];
  route_memory?: LearningSnapshot['route_memory'];
  room_map?: LearningSnapshot['room_map'];
  route_graph_file?: string;
  room_map_file?: string;
  room_map_sha256?: string;
  route_graph_sha256?: string;
  sha256?: string;
}

export interface ChampionCatalog {
  count: number;
  latest: ChampionSummary | null;
  best_completion: ChampionSummary | null;
  capture_pending: boolean;
  error?: string | null;
}


export interface MotorGuidance {
  active: boolean;
  stick: number[];
  strength: number;
  button_quiet: number;
  distance: number | null;
  source: string;
  target: number[] | null;
  blocked?: boolean;
  detour?: string | null;
  direct_probe?: string | null;
  stuck_scale?: number;
  route_active?: boolean;
  route_path_nodes?: number;
  route_confidence?: number;
  route_target_gap?: number | null;
  route_waypoint?: number[] | null;
  route_waypoint_id?: string | null;
  route_partial?: boolean;
  route_edge_key?: string | null;
  route_edge_failures?: number;
  frontier_active?: boolean;
  frontier_direction?: string | null;
  frontier_stable?: boolean;
  frontier_age_s?: number | null;
  exit_active?: boolean;
  exit_index?: number | null;
  exit_direct_reachable?: boolean;
  forced_escape?: boolean;
  remembered_escape?: boolean;
}

export interface Snapshot {
  status: string;
  reason: string;
  run_id: string | null;
  run_mode: 'train' | 'evaluation';
  active_champion: ChampionSummary | null;
  champions: ChampionCatalog;
  elapsed_s: number;
  connection: {
    game: boolean;
    source: 'soh' | 'simulator' | null;
    realtime: boolean;
    state_hz: number | null;
    ai: boolean;
    ai_state: string;
    ai_error: string;
  };
  thought: {
    summary: string;
    state: string;
    thinking_ms: number;
    intent: AgentIntent | null;
    objective_lock: {
      active: boolean;
      trackable: boolean;
      objective: string | null;
      completion: ObjectiveCompletion | null;
      completed_count: number;
      last_completion: Record<string, unknown> | null;
      replans_suppressed: number;
    };
    motor: string;
    guidance: MotorGuidance | null;
    trigger: string | null;
  };
  input: InputState;
  learning: LearningSnapshot;
  metrics: Record<string, unknown> | null;
  usage: UsageSnapshot;
  bridge: {
    connected: boolean;
    last_seen_age_s: number | null;
    realtime: Record<string, unknown>;
  };
}
