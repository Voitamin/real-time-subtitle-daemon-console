export type RuntimeStatus = {
  mode: string;
  alert_level?: "green" | "yellow" | "red";
  forced_mode?: string;
  slack_ms: number | null;
  deadline_remaining_ms?: number | null;
  risk_ratio?: number | null;
  predict_percentile?: number;
  warmup_active?: boolean;
  fast_blocked_by_latest_ready?: boolean;
  base_delay_sec?: number;
  asr_delay_sec?: number;
  delay_adjust_sec?: number;
  effective_delay_sec?: number;
  arrival_rate_lps: number;
  service_rate_lps: number;
  late_rate_recent: number;
  now_mono_ms?: number;
  latest_ready_source_key?: string | null;
  latest_ready_due_effective_mono_ms?: number | null;
  latest_ready_remaining_ms?: number | null;
  latest_ready_anomaly?: boolean;
  window_sec?: number;
  past_window_sec?: number;
  future_window_sec?: number;
  latest_editable_source_key?: string | null;
  queue?: {
    unfinished_count?: number;
    new_count?: number;
    inflight_count?: number;
    wal_size_bytes?: number;
  };
  stage?: {
    correct_p50_ms?: number;
    correct_p95_ms?: number;
    translate_p50_ms?: number;
    translate_p95_ms?: number;
    pipeline_p50_ms?: number;
    pipeline_p95_ms?: number;
  };
  window_stats?: {
    total_count: number;
    translated_count: number;
    deleted_count: number;
    displayed_count: number;
    editable_count: number;
  };
  terms_version?: {
    glossary_mtime_ns: number;
    names_mtime_ns: number;
  };
  s1_join_suppressed_count_recent?: number;
  s1_join_total_count_recent?: number;
  s1_join_ratio_recent?: number;
  translate_fast_count_recent?: number;
  translate_fast_ratio_recent?: number;
  translate_fallback_ratio_recent?: number;
  line_utilization_ratio_recent?: number;
  render_frame_count_recent?: number;
  render_second_line_used_recent?: number;
};

export type CueItem = {
  source_key: string;
  status: string;
  t_seen_mono_ms: number;
  due_mono_ms: number;
  due_effective_mono_ms: number;
  translated_mono_ms: number | null;
  displayed_at_mono_ms: number | null;
  deleted_soft: boolean;
  display_suppressed: boolean;
  join_target_source_key: string | null;
  dropped_late: boolean;
  countdown_sec: number;
  progress: number;
  editable: boolean;
  display_text: string;
  jp_raw: string;
  jp_corrected: string | null;
  zh_text: string | null;
  manual_zh_text: string | null;
  manual_locked: boolean;
  updated_by: string | null;
  revision: number;
  stage1_provider: string | null;
  stage1_model: string | null;
  stage2_provider: string | null;
  stage2_model: string | null;
  used_fast_model?: boolean;
  fallback_used: boolean;
  stage1_latency_ms: number | null;
  stage2_latency_ms: number | null;
  pipeline_latency_ms: number | null;
  last_error: string | null;
  source_kind: string;
  srt_index: number | null;
  start_ms: number | null;
  end_ms: number | null;
};

export type CuesWindowResp = {
  items: CueItem[];
  next_cursor: string | null;
  window_sec: number;
  past_window_sec?: number;
  future_window_sec?: number;
  now_mono_ms: number;
  delay_eff_ms: number;
  delay_adjust_sec?: number;
  effective_delay_sec?: number;
};

export type GlossaryItem = {
  ja: string;
  zh: string;
};

export type TermsVersion = {
  glossary_mtime_ns: number;
  names_mtime_ns: number;
};

export type GlossaryResp = {
  items: GlossaryItem[];
  version: TermsVersion;
};

export type NameItem = {
  content: string;
  sounds_like: string[];
};

export type NamesResp = {
  items: NameItem[];
  legacy_items?: string[];
  version: TermsVersion;
};
