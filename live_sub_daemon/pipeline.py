from __future__ import annotations

import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .config import AppConfig
from .knowledge import KnowledgeStore
from .llm_client import is_qwen_mt_model
from .models import CueRecord
from .runtime_control import RuntimeController
from .stage_router import StageRouter
from .state_store import StateStore


class PipelineWorker(threading.Thread):
    def __init__(
        self,
        config: AppConfig,
        state: StateStore,
        knowledge: KnowledgeStore,
        router: StageRouter,
        runtime: RuntimeController,
        stop_event: threading.Event,
        worker_id: str = "main",
        rescue: bool = False,
        rescue_index: int = 0,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        scope_freshness_sec: float = 0.0,
    ):
        super().__init__(daemon=True, name=f"pipeline-worker-{worker_id}")
        self.config = config
        self.state = state
        self.knowledge = knowledge
        self.router = router
        self.runtime = runtime
        self.stop_event = stop_event
        self.worker_id = worker_id
        self.rescue = rescue
        self.rescue_index = rescue_index
        self.allowed_source_kinds = [str(v).strip().lower() for v in (allowed_source_kinds or []) if str(v).strip()]
        self.scope_freshness_sec = max(0.0, float(scope_freshness_sec))
        self._recent_corrected_context: List[str] = []
        self._context_wait_started_mono_ms: Optional[int] = None
        self._next_inflight_reclaim_ms = 0

    def _scope_updated_after_unix(self) -> Optional[float]:
        if self.scope_freshness_sec <= 0:
            return None
        return time.time() - self.scope_freshness_sec

    def run(self) -> None:
        while not self.stop_event.is_set():
            claimed_batch: List[CueRecord] = []
            try:
                now_mono_ms = int(time.monotonic() * 1000)
                if now_mono_ms >= self._next_inflight_reclaim_ms:
                    self._next_inflight_reclaim_ms = now_mono_ms + 5000
                    retry_factor = max(1, int(self.config.llm.max_retries) + 1)
                    stale_after_ms = int(
                        max(
                            30000.0,
                            (self.config.llm.correct_timeout_sec + self.config.llm.translate_timeout_sec)
                            * 1000.0
                            * retry_factor
                            * 1.5,
                        )
                    )
                    self.state.release_stale_inflight(
                        now_mono_ms=now_mono_ms,
                        stale_after_ms=stale_after_ms,
                        allowed_source_kinds=self.allowed_source_kinds or None,
                        updated_after_unix=self._scope_updated_after_unix(),
                    )

                plan = self.runtime.get_worker_plan(rescue=self.rescue)
                if self.rescue:
                    if not self.runtime.is_red_mode() or self.rescue_index >= self.runtime.rescue_parallelism():
                        time.sleep(0.2)
                        continue

                queue_stats = self.state.fetch_new_queue_stats(
                    now_mono_ms=now_mono_ms,
                    delay_adjust_ms=self.runtime.delay_adjust_ms(),
                    allowed_source_kinds=self.allowed_source_kinds or None,
                    updated_after_unix=self._scope_updated_after_unix(),
                )
                # Policy: S1 always runs non-fast model; only S2 can use fast model.
                use_fast_translate_model, _fast_blocked = self.runtime.resolve_fast_model_usage(
                    requested_fast=plan.use_fast_model,
                    queue_stats=queue_stats,
                )
                new_count = int(queue_stats.get("new_count") or 0)
                if new_count <= 0:
                    self._context_wait_started_mono_ms = None
                    time.sleep(0.1)
                    continue

                oldest_wait_ms = int(queue_stats.get("oldest_new_wait_ms") or 0)
                should_flush = (
                    new_count >= plan.batch_max_lines
                    or oldest_wait_ms >= int(plan.batch_max_wait_sec * 1000)
                    or self.rescue
                )
                if not should_flush:
                    time.sleep(0.05)
                    continue

                mark_context_miss = False
                if not self.rescue and plan.mode == "red":
                    reserve = max(1, self.runtime.rescue_parallelism() * plan.batch_max_lines)
                    self.state.mark_oldest_new_context_miss(
                        limit=reserve,
                        allowed_source_kinds=self.allowed_source_kinds or None,
                        updated_after_unix=self._scope_updated_after_unix(),
                    )
                if not self.rescue and plan.context_window > 0:
                    oldest_seen = queue_stats.get("oldest_new_seen_mono_ms")
                    if oldest_seen is not None and self.state.has_older_inflight_than(
                        int(oldest_seen),
                        allowed_source_kinds=self.allowed_source_kinds or None,
                        updated_after_unix=self._scope_updated_after_unix(),
                    ):
                        if self._context_wait_started_mono_ms is None:
                            self._context_wait_started_mono_ms = now_mono_ms
                        waited_ms = now_mono_ms - self._context_wait_started_mono_ms
                        if waited_ms < int(plan.context_wait_timeout_sec * 1000):
                            time.sleep(0.05)
                            continue
                        self.state.mark_oldest_new_context_miss(
                            limit=plan.batch_max_lines,
                            allowed_source_kinds=self.allowed_source_kinds or None,
                            updated_after_unix=self._scope_updated_after_unix(),
                        )
                        mark_context_miss = True
                    else:
                        self._context_wait_started_mono_ms = None

                claimed_batch = self.state.fetch_and_claim_batch(
                    limit=plan.batch_max_lines,
                    owner=self.worker_id,
                    now_mono_ms=now_mono_ms,
                    contextless_only=plan.contextless_only,
                    mark_context_miss=mark_context_miss,
                    allowed_source_kinds=self.allowed_source_kinds or None,
                    updated_after_unix=self._scope_updated_after_unix(),
                )
                if not claimed_batch:
                    time.sleep(0.05)
                    continue

                self._context_wait_started_mono_ms = None
                started = time.monotonic()
                snapshot = self.knowledge.get_snapshot()
                s1_mode = plan.s1_mode
                s1_skipped = s1_mode == "off"
                display_suppressed_map: Dict[str, bool] = {}
                join_target_map: Dict[str, str] = {}

                if s1_skipped:
                    correct_route = None
                    correct_result = None
                    corrected_texts = _default_stage1_texts(claimed_batch)
                    self.state.record_metric(
                        stage="correct_skipped",
                        batch_size=len(claimed_batch),
                        latency_ms=0,
                        ok=True,
                        timed_out=False,
                        cached_tokens=0,
                    )
                    self.runtime.observe_s1_skipped(count=len(claimed_batch), now_mono_ms=now_mono_ms)
                else:
                    if s1_mode == "lite":
                        correct_prompt = build_correct_prompt_lite(
                            snapshot.glossary,
                            snapshot.names_whitelist,
                            glossary_limit=self.config.knowledge.correct_glossary_limit,
                            names_limit=self.config.knowledge.names_limit,
                            punctuation_only=plan.s1_lite_punctuation_only,
                        )
                    elif s1_mode == "lite_ai_join":
                        correct_prompt = build_correct_prompt_lite_ai_join(
                            snapshot.glossary,
                            snapshot.names_whitelist,
                            glossary_limit=self.config.knowledge.correct_glossary_limit,
                            names_limit=self.config.knowledge.names_limit,
                            marker=self.config.pipeline.s1_ai_join_marker,
                            allow_light_compress=self.config.pipeline.s1_ai_join_allow_light_compress,
                        )
                    else:
                        correct_prompt = build_correct_prompt(
                            snapshot.glossary,
                            snapshot.names_whitelist,
                            glossary_limit=self.config.knowledge.correct_glossary_limit,
                            names_limit=self.config.knowledge.names_limit,
                        )
                    correct_items = [
                        {"source_key": cue.source_key, "text": _cue_stage1_input(cue)}
                        for cue in claimed_batch
                    ]

                    correct_route = self.router.run_correct_batch(
                        stage_name="jp_correct",
                        system_prompt=correct_prompt,
                        items=correct_items,
                        max_retries=plan.max_retries,
                        retry_backoff_sec=self.config.llm.retry_backoff_sec,
                        temperature=self.config.llm.temperature,
                        use_fast_model=False,
                    )
                    correct_result = correct_route.result
                    self.state.record_metric(
                        stage="correct",
                        batch_size=len(claimed_batch),
                        latency_ms=correct_result.latency_ms,
                        ok=correct_result.ok,
                        timed_out=correct_result.timed_out,
                        cached_tokens=correct_result.cached_tokens,
                    )

                    corrected_texts = _merge_with_fallback(claimed_batch, correct_result.outputs)
                    if s1_mode == "lite_ai_join":
                        corrected_texts, display_suppressed_map, join_target_map = _apply_ai_join_plan(
                            cues=claimed_batch,
                            corrected_texts=corrected_texts,
                            marker=self.config.pipeline.s1_ai_join_marker,
                            max_chain=self.config.pipeline.s1_ai_join_max_chain,
                            keep_min_chars=self.config.pipeline.s1_ai_join_keep_min_chars,
                        )
                        self.runtime.observe_s1_join(
                            total_count=len(claimed_batch),
                            suppressed_count=sum(1 for v in display_suppressed_map.values() if v),
                            now_mono_ms=int(time.monotonic() * 1000),
                        )

                eligible_cues = [cue for cue in claimed_batch if not display_suppressed_map.get(cue.source_key, False)]
                if not eligible_cues:
                    eligible_cues = list(claimed_batch)

                translate_provider_runtime = self.config.llm.provider_settings[self.config.llm.translate_provider]
                translate_model_for_prompt = (
                    translate_provider_runtime.fast_translate_model
                    if use_fast_translate_model and translate_provider_runtime.fast_translate_model
                    else translate_provider_runtime.translate_model
                )
                use_qwen_mt = is_qwen_mt_model(translate_model_for_prompt)
                translation_options = None
                if use_qwen_mt:
                    translate_prompt = ""
                    translation_options = build_qwen_mt_translation_options(
                        snapshot.glossary,
                        snapshot.names_whitelist,
                        glossary_limit=self.config.knowledge.translate_glossary_limit,
                        names_limit=self.config.knowledge.names_limit,
                    )
                else:
                    translate_prompt = build_translate_prompt(
                        snapshot.glossary,
                        snapshot.names_whitelist,
                        glossary_limit=self.config.knowledge.translate_glossary_limit,
                        names_limit=self.config.knowledge.names_limit,
                    )

                context_window = 0 if (use_qwen_mt or plan.context_window <= 0 or self.rescue) else plan.context_window
                rolling_context = list(self._recent_corrected_context)
                translate_items = []
                for cue in eligible_cues:
                    current_text = corrected_texts[cue.source_key]
                    context_lines = rolling_context[-context_window:] if context_window > 0 else []
                    translated_input = (
                        current_text
                        if use_qwen_mt
                        else _compose_translate_input(current_text=current_text, context_lines=context_lines)
                    )
                    translate_items.append({"source_key": cue.source_key, "text": translated_input})
                    rolling_context.append(current_text)
                if context_window > 0:
                    self._recent_corrected_context = rolling_context[-context_window:]

                translate_route = self.router.run_translate_batch(
                    stage_name="jp_to_zh",
                    system_prompt=translate_prompt,
                    items=translate_items,
                    max_retries=plan.max_retries,
                    retry_backoff_sec=self.config.llm.retry_backoff_sec,
                    temperature=self.config.llm.temperature,
                    use_fast_model=use_fast_translate_model,
                    translation_options=translation_options,
                )
                translate_result = translate_route.final.result
                stage2_route_latency_ms = translate_route.primary.result.latency_ms + (
                    translate_route.fallback.result.latency_ms if translate_route.fallback else 0
                )
                self.state.record_metric(
                    stage="translate",
                    batch_size=len(eligible_cues),
                    latency_ms=stage2_route_latency_ms,
                    ok=translate_result.ok,
                    timed_out=translate_result.timed_out,
                    cached_tokens=translate_result.cached_tokens,
                )
                if translate_route.fallback is not None:
                    self.state.record_metric(
                        stage="translate_fallback",
                        batch_size=len(eligible_cues),
                        latency_ms=translate_route.fallback.result.latency_ms,
                        ok=translate_route.fallback.result.ok,
                        timed_out=translate_route.fallback.result.timed_out,
                        cached_tokens=translate_route.fallback.result.cached_tokens,
                    )
                self.runtime.observe_translate_route(
                    primary_ok=translate_route.primary.result.ok,
                    fallback_used=translate_route.fallback_used,
                    fallback_ok=bool(translate_route.fallback and translate_route.fallback.result.ok),
                    used_fast_model=use_fast_translate_model,
                    now_mono_ms=int(time.monotonic() * 1000),
                )

                translated_at = int(time.monotonic() * 1000)
                pipeline_latency = int((time.monotonic() - started) * 1000)
                self.state.record_metric(
                    stage="pipeline_total",
                    batch_size=len(claimed_batch),
                    latency_ms=pipeline_latency,
                    ok=translate_result.ok,
                    timed_out=translate_result.timed_out,
                    cached_tokens=None,
                )
                error_parts: List[str] = []
                if not translate_route.primary.result.ok and translate_route.primary.result.error:
                    error_parts.append(f"primary:{translate_route.primary.result.error}")
                if translate_route.fallback is not None and not translate_route.fallback.result.ok:
                    if translate_route.fallback.result.error:
                        error_parts.append(f"fallback:{translate_route.fallback.result.error}")
                if not error_parts and not translate_result.ok and translate_result.error:
                    error_parts.append(translate_result.error)
                error_message = "; ".join(error_parts) if error_parts else None
                self.state.save_pipeline_results(
                    cues=claimed_batch,
                    corrected_texts=corrected_texts,
                    translated_texts=translate_result.outputs,
                    translated_at_mono_ms=translated_at,
                    fallback_mode=self.config.fallback.mode,
                    llm_latency_ms=pipeline_latency,
                    error_message=error_message,
                    stage1_provider=None if correct_route is None else correct_route.provider,
                    stage1_model=None if correct_route is None else correct_route.model,
                    stage2_provider=translate_route.final.provider,
                    stage2_model=translate_route.final.model,
                    fallback_used=translate_route.fallback_used,
                    stage1_latency_ms=0 if correct_result is None else correct_result.latency_ms,
                    stage2_latency_ms=stage2_route_latency_ms,
                    s1_skipped=s1_skipped,
                    display_suppressed_map=display_suppressed_map,
                    join_target_map=join_target_map,
                )

                late_count = sum(1 for cue in eligible_cues if translated_at > cue.due_mono_ms)
                self.runtime.observe_completion(
                    total_count=len(claimed_batch),
                    late_count=late_count,
                    stage1_ms=0 if correct_result is None else correct_result.latency_ms,
                    stage2_ms=stage2_route_latency_ms,
                    pipeline_ms=pipeline_latency,
                    now_mono_ms=translated_at,
                )
                claimed_batch = []
            except Exception as exc:  # noqa: BLE001
                if claimed_batch:
                    self.state.release_claimed_batch(
                        [cue.source_key for cue in claimed_batch],
                        error_message=f"worker_error:{exc}",
                    )
                print(f"[pipeline:{self.worker_id}] worker error: {exc}", file=sys.stderr)
                time.sleep(0.3)


def _merge_with_fallback(cues: List[CueRecord], outputs: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for cue in cues:
        text = outputs.get(cue.source_key)
        if text is None:
            text = _cue_stage1_input(cue)
        merged[cue.source_key] = text
    return merged


def _default_stage1_texts(cues: List[CueRecord]) -> Dict[str, str]:
    return {cue.source_key: _cue_stage1_input(cue) for cue in cues}


def _cue_stage1_input(cue: CueRecord) -> str:
    if cue.jp_canonicalized:
        return cue.jp_canonicalized
    if cue.jp_aggregated:
        return cue.jp_aggregated
    return cue.jp_raw


def _parse_join_marker(text: str, marker: str) -> Tuple[bool, str]:
    raw = str(text or "").strip()
    token = str(marker or "").strip()
    if not token:
        return False, raw
    if not raw.startswith(token):
        return False, raw
    stripped = raw[len(token):].lstrip()
    return True, stripped


def _concat_join_text(left: str, right: str) -> str:
    left_clean = left.strip()
    right_clean = right.strip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    if left_clean[-1].isascii() and left_clean[-1].isalnum() and right_clean[0].isascii() and right_clean[0].isalnum():
        return f"{left_clean} {right_clean}"
    return f"{left_clean}{right_clean}"


def _apply_ai_join_plan(
    *,
    cues: List[CueRecord],
    corrected_texts: Dict[str, str],
    marker: str,
    max_chain: int,
    keep_min_chars: int,
) -> Tuple[Dict[str, str], Dict[str, bool], Dict[str, str]]:
    normalized: Dict[str, str] = {}
    flags: Dict[str, bool] = {}
    suppressed: Dict[str, bool] = {}
    targets: Dict[str, str] = {}
    max_chain = max(0, int(max_chain))
    keep_min_chars = max(0, int(keep_min_chars))

    for cue in cues:
        key = cue.source_key
        join_flag, text = _parse_join_marker(corrected_texts.get(key, _cue_stage1_input(cue)), marker)
        text = text.strip()
        if not text:
            text = _cue_stage1_input(cue)
            join_flag = False
        normalized[key] = text
        flags[key] = join_flag
        suppressed[key] = False

    chain_len = 0
    for idx, cue in enumerate(cues):
        key = cue.source_key
        join_flag = flags.get(key, False)
        can_join = idx + 1 < len(cues)

        if join_flag and keep_min_chars > 0 and len(normalized[key]) >= keep_min_chars:
            join_flag = False
        if join_flag and max_chain > 0 and chain_len >= max_chain:
            join_flag = False
        if not join_flag or not can_join:
            chain_len = 0
            continue

        next_key = cues[idx + 1].source_key
        normalized[next_key] = _concat_join_text(normalized[key], normalized[next_key])
        suppressed[key] = True
        targets[key] = next_key
        chain_len += 1

    return normalized, suppressed, targets


def build_correct_prompt(
    glossary: Dict[str, str],
    names_whitelist: List[str],
    glossary_limit: int,
    names_limit: int,
) -> str:
    glossary_lines = "\n".join(f"- {ja} => {zh}" for ja, zh in list(glossary.items())[: max(0, glossary_limit)])
    names_lines = "\n".join(f"- {name}" for name in names_whitelist[: max(0, names_limit)])
    return (
        "You are a Japanese subtitle correction engine.\n"
        "Task: JP->JP correction only.\n"
        "Rules:\n"
        "1) Do not add or remove information.\n"
        "2) Keep person names / IDs / song names unchanged unless they match whitelist corrections.\n"
        "3) Fix homophone mistakes, punctuation, and sentence boundaries for natural spoken Japanese.\n"
        "4) If a token is clearly a game term (move name, character name, in-game term), preserve it even if it looks like a typo, unless glossary explicitly requires correction.\n"
        "5) Output strictly JSON array of {source_key, text}.\n"
        "Name whitelist:\n"
        f"{names_lines or '- (empty)'}\n"
        "Glossary hints (do not translate in this stage):\n"
        f"{glossary_lines or '- (empty)'}"
    )


def build_correct_prompt_lite(
    glossary: Dict[str, str],
    names_whitelist: List[str],
    glossary_limit: int,
    names_limit: int,
    punctuation_only: bool,
) -> str:
    glossary_lines = "\n".join(f"- {ja} => {zh}" for ja, zh in list(glossary.items())[: max(0, glossary_limit)])
    names_lines = "\n".join(f"- {name}" for name in names_whitelist[: max(0, names_limit)])
    action_line = (
        "3) Only normalize punctuation and sentence boundaries; keep lexical tokens unchanged unless they are obvious ASR noise."
        if punctuation_only
        else "3) Normalize punctuation and light spoken disfluencies without changing semantic content."
    )
    return (
        "You are a Japanese subtitle normalization engine.\n"
        "Task: JP->JP normalization only.\n"
        "Rules:\n"
        "1) Do not add, remove, summarize, or rephrase meaning.\n"
        "2) Never change names/IDs/song titles already matching whitelist or glossary.\n"
        f"{action_line}\n"
        "4) Keep fragmentary lines fragmentary; do not complete missing text.\n"
        "5) Keep game terms intact unless glossary explicitly maps correction.\n"
        "6) Output strictly JSON array of {source_key, text}.\n"
        "Name whitelist:\n"
        f"{names_lines or '- (empty)'}\n"
        "Glossary hints:\n"
        f"{glossary_lines or '- (empty)'}"
    )


def build_correct_prompt_lite_ai_join(
    glossary: Dict[str, str],
    names_whitelist: List[str],
    glossary_limit: int,
    names_limit: int,
    marker: str,
    allow_light_compress: bool,
) -> str:
    glossary_lines = "\n".join(f"- {ja} => {zh}" for ja, zh in list(glossary.items())[: max(0, glossary_limit)])
    names_lines = "\n".join(f"- {name}" for name in names_whitelist[: max(0, names_limit)])
    compress_line = (
        "5) You may lightly compress empty fillers/repetitions (e.g. はい, えっと, あの, repeated backchannels) if no factual loss."
        if allow_light_compress
        else "5) Do not compress words; only decide joining."
    )
    return (
        "You are a Japanese subtitle normalization-and-joining engine.\n"
        "Task: JP->JP normalization with optional semantic line joining.\n"
        "Output contract:\n"
        "- Return strict JSON array of {source_key, text}.\n"
        f"- If one line should be merged into the NEXT line, prefix text with '{marker}'.\n"
        "- Only prefix marker for merge-to-next. No other tags.\n"
        "Rules:\n"
        "1) Do not add, remove, or invent facts.\n"
        "2) Keep names/IDs/song titles and game terms stable by whitelist/glossary.\n"
        "3) Prefer natural spoken sentence rhythm, reduce overly fragmented short lines.\n"
        "4) Keep fragmentary lines fragmentary if they cannot be safely merged.\n"
        f"{compress_line}\n"
        "6) Keep source_key unchanged and output item count equal to input count.\n"
        "Name whitelist:\n"
        f"{names_lines or '- (empty)'}\n"
        "Glossary hints:\n"
        f"{glossary_lines or '- (empty)'}"
    )


def build_translate_prompt(
    glossary: Dict[str, str],
    names_whitelist: List[str],
    glossary_limit: int,
    names_limit: int,
) -> str:
    glossary_lines = "\n".join(f"- {ja} => {zh}" for ja, zh in list(glossary.items())[: max(0, glossary_limit)])
    names_lines = "\n".join(f"- {name}" for name in names_whitelist[: max(0, names_limit)])
    return (
        "You are a Japanese-to-Chinese livestream subtitle translator.\n"
        "Task: translate JP->ZH for real-time esports commentary captions.\n"
        "Output style goal: natural spoken Chinese, concise, easy to read on stream.\n"
        "Workflow:\n"
        "A) First read the whole batch as one coherent block.\n"
        "B) Resolve references and incomplete starts using nearby lines in the same batch.\n"
        "C) Then output one Chinese line per source_key in original order.\n"
        "Input format note:\n"
        "- each item's `text` may include [CONTEXT] and [CURRENT] sections.\n"
        "- use [CONTEXT] only for reference.\n"
        "- translate only the [CURRENT] sentence.\n"
        "Rules:\n"
        "1) Prioritize natural Chinese phrasing over literal translation.\n"
        "2) Keep subtitle text short and readable, but do not over-compress key meaning.\n"
        "3) Keep terminology consistent with glossary.\n"
        "4) Person names / song names stay in original form unless glossary defines translation.\n"
        "5) Do not add facts or meaning not present in source.\n"
        "6) Filler words and backchannels (e.g., はい, えっと, あの) may be merged or omitted when they carry no meaning.\n"
        "7) Avoid literal machine phrasing; rewrite to idiomatic Chinese broadcast speech.\n"
        "8) Prefer punctuation and rhythm that fit spoken captions.\n"
        "9) If a line is clearly fragmentary/truncated, use ellipsis '……' or '(疑似)' to mark uncertainty, and do not complete missing parts.\n"
        "10) Output strictly JSON array of {source_key, text} only.\n"
        "Name whitelist:\n"
        f"{names_lines or '- (empty)'}\n"
        "Glossary:\n"
        f"{glossary_lines or '- (empty)'}"
    )


def build_qwen_mt_translation_options(
    glossary: Dict[str, str],
    names_whitelist: List[str],
    glossary_limit: int,
    names_limit: int,
) -> Dict[str, object]:
    tm_map: Dict[str, str] = {}
    for ja, zh in list(glossary.items())[: max(0, glossary_limit)]:
        if ja and zh:
            tm_map[ja] = zh
    for name in names_whitelist[: max(0, names_limit)]:
        if name and name not in tm_map:
            tm_map[name] = name
    tm_list = [{"source": source, "target": target} for source, target in tm_map.items()]
    return {"source_lang": "Japanese", "target_lang": "Chinese", "tm_list": tm_list}


def _compose_translate_input(current_text: str, context_lines: List[str]) -> str:
    if not context_lines:
        return current_text
    context_block = "\n".join(f"- {line}" for line in context_lines)
    return (
        "[CONTEXT]\n"
        f"{context_block}\n"
        "[CURRENT]\n"
        f"{current_text}"
    )

