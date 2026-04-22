from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class StageResult:
    outputs: Dict[str, str]
    latency_ms: int
    ok: bool
    timed_out: bool
    error: Optional[str]
    cached_tokens: Optional[int]


def is_qwen_mt_model(model: str) -> bool:
    return str(model).strip().lower().startswith("qwen-mt")


class XAIClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def run_batch(
        self,
        model: str,
        stage_name: str,
        system_prompt: str,
        items: Sequence[dict[str, str]],
        timeout_sec: float,
        max_retries: int,
        retry_backoff_sec: float,
        temperature: float,
        translation_options: Optional[dict[str, Any]] = None,
    ) -> StageResult:
        if is_qwen_mt_model(model):
            return self._run_batch_qwen_mt(
                model=model,
                items=items,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
                translation_options=translation_options,
            )

        last_error: Optional[str] = None
        last_timed_out = False
        start = time.monotonic()

        for attempt in range(max_retries + 1):
            try:
                result = self._send_once(
                    model=model,
                    stage_name=stage_name,
                    system_prompt=system_prompt,
                    items=items,
                    timeout_sec=timeout_sec,
                    temperature=temperature,
                )
                total_latency_ms = int((time.monotonic() - start) * 1000)
                return StageResult(
                    outputs=result.outputs,
                    latency_ms=total_latency_ms,
                    ok=True,
                    timed_out=False,
                    error=None,
                    cached_tokens=result.cached_tokens,
                )
            except TimeoutError as exc:
                last_error = str(exc)
                last_timed_out = True
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                last_timed_out = False

            if attempt < max_retries:
                time.sleep(retry_backoff_sec)

        total_latency_ms = int((time.monotonic() - start) * 1000)
        return StageResult(
            outputs={},
            latency_ms=total_latency_ms,
            ok=False,
            timed_out=last_timed_out,
            error=last_error,
            cached_tokens=None,
        )

    def _run_batch_qwen_mt(
        self,
        model: str,
        items: Sequence[dict[str, str]],
        timeout_sec: float,
        max_retries: int,
        retry_backoff_sec: float,
        translation_options: Optional[dict[str, Any]],
    ) -> StageResult:
        start = time.monotonic()
        outputs: Dict[str, str] = {}
        cached_tokens_total = 0
        has_cached_tokens = False
        last_error: Optional[str] = None
        last_timed_out = False

        options = _normalize_translation_options(translation_options)
        for item in items:
            source_key = item.get("source_key")
            text = item.get("text")
            if not isinstance(source_key, str) or not isinstance(text, str):
                last_error = "invalid item: each item must contain string source_key and text"
                total_latency_ms = int((time.monotonic() - start) * 1000)
                return StageResult(
                    outputs=outputs,
                    latency_ms=total_latency_ms,
                    ok=False,
                    timed_out=False,
                    error=last_error,
                    cached_tokens=(cached_tokens_total if has_cached_tokens else None),
                )

            sent = False
            for attempt in range(max_retries + 1):
                try:
                    translated_text, cached_tokens = self._send_qwen_mt_once(
                        model=model,
                        text=text,
                        timeout_sec=timeout_sec,
                        translation_options=options,
                    )
                    outputs[source_key] = translated_text
                    if isinstance(cached_tokens, int):
                        cached_tokens_total += cached_tokens
                        has_cached_tokens = True
                    sent = True
                    break
                except TimeoutError as exc:
                    last_error = str(exc)
                    last_timed_out = True
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    last_timed_out = False

                if attempt < max_retries:
                    time.sleep(retry_backoff_sec)

            if not sent:
                total_latency_ms = int((time.monotonic() - start) * 1000)
                return StageResult(
                    outputs=outputs,
                    latency_ms=total_latency_ms,
                    ok=False,
                    timed_out=last_timed_out,
                    error=last_error,
                    cached_tokens=(cached_tokens_total if has_cached_tokens else None),
                )

        total_latency_ms = int((time.monotonic() - start) * 1000)
        return StageResult(
            outputs=outputs,
            latency_ms=total_latency_ms,
            ok=True,
            timed_out=False,
            error=None,
            cached_tokens=(cached_tokens_total if has_cached_tokens else None),
        )

    def _send_once(
        self,
        model: str,
        stage_name: str,
        system_prompt: str,
        items: Sequence[dict[str, str]],
        timeout_sec: float,
        temperature: float,
    ) -> StageResult:
        endpoint = f"{self.base_url}/chat/completions"
        user_payload = {
            "stage": stage_name,
            "items": list(items),
            "output_contract": [
                {
                    "source_key": "same as input source_key",
                    "text": "processed text",
                }
            ],
            "notes": "Return only JSON with no markdown fences.",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": temperature,
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "live-sub-daemon/0.1",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                resp_body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise TimeoutError(f"request timeout or network error: {exc}") from exc

        parsed = json.loads(resp_body)
        message = _extract_message_text(parsed)
        outputs = _parse_output_items(message)

        cached_tokens = None
        usage = parsed.get("usage")
        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                value = details.get("cached_tokens")
                if isinstance(value, int):
                    cached_tokens = value

        return StageResult(
            outputs=outputs,
            latency_ms=0,
            ok=True,
            timed_out=False,
            error=None,
            cached_tokens=cached_tokens,
        )

    def _send_qwen_mt_once(
        self,
        model: str,
        text: str,
        timeout_sec: float,
        translation_options: dict[str, Any],
    ) -> Tuple[str, Optional[int]]:
        endpoint = f"{self.base_url}/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "translation_options": translation_options,
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "live-sub-daemon/0.1",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                resp_body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise TimeoutError(f"request timeout or network error: {exc}") from exc

        parsed = json.loads(resp_body)
        message = _extract_message_text(parsed).strip()
        if not message:
            raise ValueError("Model output is empty")

        cached_tokens = None
        usage = parsed.get("usage")
        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                value = details.get("cached_tokens")
                if isinstance(value, int):
                    cached_tokens = value
        return message, cached_tokens


def _normalize_translation_options(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "source_lang": "Japanese",
        "target_lang": "Chinese",
    }
    if isinstance(raw, dict):
        options.update(raw)

    tm_raw = options.get("tm_list")
    if not isinstance(tm_raw, list):
        options["tm_list"] = []
        return options

    tm_list: List[dict[str, str]] = []
    for item in tm_raw:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        target = item.get("target")
        if isinstance(source, str) and isinstance(target, str) and source and target:
            tm_list.append({"source": source, "target": target})
    options["tm_list"] = tm_list
    return options


def _extract_message_text(parsed: dict[str, Any]) -> str:
    message_obj = (
        parsed.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if isinstance(message_obj, list):
        parts: List[str] = []
        for part in message_obj:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    message = str(message_obj).strip()
    if message:
        return message

    fallback = parsed.get("output_text")
    if isinstance(fallback, str):
        return fallback
    return ""


def _parse_output_items(content: str) -> Dict[str, str]:
    content = content.strip()
    payload = None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None

    if payload is None:
        payload = _extract_json_payload(content)

    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Model output is not a JSON list")

    out: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        source_key = item.get("source_key")
        text = item.get("text")
        if isinstance(source_key, str) and isinstance(text, str):
            out[source_key] = text
    if not out:
        raise ValueError("Model output contains no valid items")
    return out


def _extract_json_payload(content: str):
    matches = re.findall(r"\{[\s\S]*\}|\[[\s\S]*\]", content)
    for raw in matches:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    raise ValueError("Unable to extract JSON payload from model output")
