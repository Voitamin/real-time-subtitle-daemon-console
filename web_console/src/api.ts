import type { CuesWindowResp, GlossaryResp, NamesResp, RuntimeStatus } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  return (await resp.json()) as T;
}

export function fetchStatus(): Promise<RuntimeStatus> {
  return request<RuntimeStatus>("/api/status");
}

export function fetchCuesWindow(windowSec: number, limit = 500, cursor?: string): Promise<CuesWindowResp> {
  const qs = new URLSearchParams();
  qs.set("window_sec", String(windowSec));
  qs.set("limit", String(limit));
  if (cursor) qs.set("cursor", cursor);
  return request<CuesWindowResp>(`/api/cues/window?${qs.toString()}`);
}

export function patchCueText(sourceKey: string, text: string, revision: number): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/api/cues/${encodeURIComponent(sourceKey)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, revision }),
  });
}

export function deleteCue(sourceKey: string, revision: number): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/api/cues/${encodeURIComponent(sourceKey)}/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision }),
  });
}

export function restoreCue(sourceKey: string, revision: number): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/api/cues/${encodeURIComponent(sourceKey)}/restore`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision }),
  });
}

export function fetchGlossary(): Promise<GlossaryResp> {
  return request<GlossaryResp>("/api/terms/glossary");
}

export function upsertGlossary(ja: string, zh: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/api/terms/glossary/upsert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ja, zh }),
  });
}

export function deleteGlossary(ja: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/api/terms/glossary/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ja }),
  });
}

export function fetchNames(): Promise<NamesResp> {
  return request<NamesResp>("/api/terms/names");
}

export function upsertName(content: string, soundsLike: string[], prevContent?: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/api/terms/names/upsert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, sounds_like: soundsLike, prev_content: prevContent }),
  });
}

export function addName(name: string): Promise<{ ok: boolean }> {
  return upsertName(name, []);
}

export function deleteName(content: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/api/terms/names/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
}

export function adjustDelay(deltaSec: number): Promise<{ ok: boolean; delay_adjust_sec: number; effective_delay_sec: number }> {
  return request<{ ok: boolean; delay_adjust_sec: number; effective_delay_sec: number }>("/api/control/delay_adjust", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ delta_sec: deltaSec }),
  });
}

export function resetDelayAdjust(): Promise<{ ok: boolean; delay_adjust_sec: number; effective_delay_sec: number }> {
  return request<{ ok: boolean; delay_adjust_sec: number; effective_delay_sec: number }>("/api/control/delay_reset", {
    method: "POST",
  });
}

export function jumpToLatest(): Promise<{ ok: boolean; skipped_count: number; timestamp_mono_ms: number }> {
  return request<{ ok: boolean; skipped_count: number; timestamp_mono_ms: number }>("/api/control/jump_to_latest", {
    method: "POST",
  });
}

export function flushPending(): Promise<{ ok: boolean; flushed_count: number; timestamp_mono_ms: number }> {
  return request<{ ok: boolean; flushed_count: number; timestamp_mono_ms: number }>("/api/control/flush_pending", {
    method: "POST",
  });
}
