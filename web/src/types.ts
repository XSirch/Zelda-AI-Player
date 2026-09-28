export interface AgentIntent {
  objective: string;
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
  achievements?: LearningAchievement[];
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
  sha256?: string;
}

export interface ChampionCatalog {
  count: number;
  latest: ChampionSummary | null;
  best_completion: ChampionSummary | null;
  capture_pending: boolean;
  error?: string | null;
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
    motor: string;
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
