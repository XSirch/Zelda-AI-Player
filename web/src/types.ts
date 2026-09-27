export interface AgentIntent {
  objective: string;
  summary: string;
  mode: 'explore' | 'navigate' | 'interact' | 'combat' | 'dialogue' | 'menu' | 'observe';
  target_actor_id: number | null;
  target_actor_params: number | null;
  target_actor_uid: string | null;
  target_position: number[] | null;
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

export interface UsageState {
  provider: string | null;
  model: string | null;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  reasoning_output_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
  known_cost_usd: number;
  quota: {
    available?: boolean;
    connected?: boolean;
    plan?: string | null;
    windows?: QuotaWindow[];
    credits?: { has_credits: boolean; unlimited: boolean; balance: string | null } | null;
    individual_limit?: { limit: string | null; used: string | null; remaining_percent: number | null; resets_at: number | null } | null;
    limit?: number | null;
    limit_remaining?: number | null;
    error?: string;
  };
  quota_updated_at: number | null;
}

export interface Snapshot {
  status: string;
  reason: string;
  run_id: string | null;
  elapsed_s: number;
  connection: {
    game: boolean;
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
  };
  input: InputState;
  learning: Record<string, unknown>;
  metrics: Record<string, unknown> | null;
  usage: UsageSnapshot;
  usage: UsageState;
  bridge: {
    connected: boolean;
    last_seen_age_s: number | null;
    realtime: Record<string, unknown>;
  };
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
  windows: QuotaWindow[];
  credits?: { has_credits: boolean; unlimited: boolean; balance: string | number | null } | null;
  individual_limit?: { limit: number | null; used: number | null; remaining_percent: number | null; resets_at: number | null } | null;
  rate_limit_reached_type?: string | null;
  error?: string | null;
}

export interface UsageSnapshot {
  provider: string | null;
  model: string | null;
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
