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
  bridge: {
    connected: boolean;
    last_seen_age_s: number | null;
    realtime: Record<string, unknown>;
  };
}
