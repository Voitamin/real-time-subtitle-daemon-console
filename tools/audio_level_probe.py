from __future__ import annotations

import argparse
import audioop
import collections
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_sub_daemon.config import _parse_toml_file  # noqa: PLC2701


def _nested_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime audio level probe for Speechmatics capture command")
    parser.add_argument("--config", default="config.toml", help="Path to TOML config")
    parser.add_argument("--capture-cmd", default=None, help="Override capture command")
    parser.add_argument("--sample-rate", type=int, default=None, help="Override sample rate")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size (bytes)")
    parser.add_argument("--interval-sec", type=float, default=0.5, help="Print interval")
    parser.add_argument("--duration-sec", type=float, default=None, help="Stop after N seconds")
    parser.add_argument("--silence-rms-threshold", type=int, default=100, help="RMS below this counts as silence")
    parser.add_argument("--bar-width", type=int, default=32, help="ASCII bar width")
    parser.add_argument("--list-dshow-devices", action="store_true", help="Print dshow devices and exit")
    return parser.parse_args()


def _list_dshow_devices() -> int:
    cmd = 'ffmpeg -hide_banner -f dshow -list_devices true -i dummy'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    print(output.strip())
    return proc.returncode


def _stderr_pump(stderr, stop_event: threading.Event) -> None:
    try:
        while not stop_event.is_set():
            line = stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                print(f"[ffmpeg] {text}")
    except Exception:
        return


def _make_bar(level: float, width: int) -> str:
    clamped = max(0.0, min(1.0, level))
    n = int(round(clamped * width))
    return "#" * n + "-" * (width - n)


def main() -> int:
    args = _parse_args()
    if args.list_dshow_devices:
        return _list_dshow_devices()

    config_data = _parse_toml_file(Path(args.config))
    sample_rate = int(
        args.sample_rate
        or _nested_get(config_data, "source.speechmatics.sample_rate", 16000)
    )
    chunk_size = int(
        args.chunk_size
        or _nested_get(config_data, "source.speechmatics.chunk_size", 4096)
    )
    capture_cmd_template = str(
        args.capture_cmd
        or _nested_get(
            config_data,
            "source.speechmatics.capture_cmd",
            "ffmpeg -hide_banner -loglevel warning -f dshow -i audio=\"CABLE Output (VB-Audio Virtual Cable)\" -ac 1 -ar {sample_rate} -f s16le -",
        )
    )
    capture_cmd = capture_cmd_template.format(sample_rate=sample_rate, chunk_size=chunk_size)

    print("audio level probe")
    print(f"config: {args.config}")
    print(f"sample_rate: {sample_rate}")
    print(f"chunk_size: {chunk_size}")
    print(f"capture_cmd: {capture_cmd}")
    print("press Ctrl+C to stop")
    print("")

    proc = subprocess.Popen(
        capture_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None:
        print("error: capture stdout unavailable")
        return 2

    stop_event = threading.Event()
    stderr_thread: Optional[threading.Thread] = None
    if proc.stderr is not None:
        stderr_thread = threading.Thread(
            target=_stderr_pump,
            args=(proc.stderr, stop_event),
            daemon=True,
            name="audio-probe-stderr",
        )
        stderr_thread.start()

    start = time.monotonic()
    last_print = start
    silence_start: Optional[float] = None
    rms_window: Deque[int] = collections.deque(maxlen=10)
    peak_window: Deque[int] = collections.deque(maxlen=10)

    try:
        while True:
            if args.duration_sec is not None and time.monotonic() - start >= args.duration_sec:
                break

            if proc.poll() is not None:
                print(f"capture process exited with code {proc.returncode}")
                return 3

            chunk = proc.stdout.read(chunk_size)
            if not chunk:
                print("no audio bytes received from capture process")
                return 4

            rms = int(audioop.rms(chunk, 2))
            peak = int(audioop.max(chunk, 2))
            rms_window.append(rms)
            peak_window.append(peak)

            now = time.monotonic()
            if rms < args.silence_rms_threshold:
                if silence_start is None:
                    silence_start = now
            else:
                silence_start = None

            if now - last_print >= max(0.1, args.interval_sec):
                avg_rms = sum(rms_window) / max(1, len(rms_window))
                avg_peak = sum(peak_window) / max(1, len(peak_window))
                rms_level = min(1.0, avg_rms / 5000.0)
                peak_level = min(1.0, avg_peak / 20000.0)
                silence_sec = 0.0 if silence_start is None else now - silence_start
                rms_bar = _make_bar(rms_level, args.bar_width)
                peak_bar = _make_bar(peak_level, args.bar_width)
                print(
                    f"t+{now-start:7.1f}s | rms={avg_rms:7.1f} [{rms_bar}] "
                    f"| peak={avg_peak:7.1f} [{peak_bar}] | silence={silence_sec:5.1f}s"
                )
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

