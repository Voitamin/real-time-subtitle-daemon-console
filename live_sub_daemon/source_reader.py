from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional

from .models import SourceCue
from .source_base import BaseSourceReader

SRT_TS_RE = re.compile(r"^(\d\d):(\d\d):(\d\d),(\d\d\d)$")


class FileSourceReader(BaseSourceReader):
    def __init__(self, srt_path: Path, txt_path: Path):
        self.srt_path = srt_path
        self.txt_path = txt_path

    def poll(self) -> List[SourceCue]:
        srt_cues = self._try_read_srt()
        if srt_cues:
            return srt_cues
        return self._read_txt_fallback()

    def _try_read_srt(self) -> Optional[List[SourceCue]]:
        if not self.srt_path.exists():
            return []
        try:
            text = self.srt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None
        cues: List[SourceCue] = []
        blocks = re.split(r"\r?\n\s*\r?\n", text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
            if len(lines) < 3:
                continue
            if not lines[0].isdigit():
                continue
            index = int(lines[0])
            if "-->" not in lines[1]:
                continue
            start_raw, end_raw = [part.strip() for part in lines[1].split("-->", 1)]
            start_ms = _parse_srt_ts_ms(start_raw)
            end_ms = _parse_srt_ts_ms(end_raw)
            if start_ms is None or end_ms is None:
                continue
            jp_raw = "\n".join(lines[2:]).strip()
            if not jp_raw:
                continue
            source_key = f"srt:{index}:{start_ms}:{end_ms}"
            cues.append(
                SourceCue(
                    source_key=source_key,
                    source_kind="srt",
                    srt_index=index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    jp_raw=jp_raw,
                )
            )
        return cues

    def _read_txt_fallback(self) -> List[SourceCue]:
        if not self.txt_path.exists():
            return []
        text = self.txt_path.read_text(encoding="utf-8", errors="replace")
        lines = [(idx, line.strip()) for idx, line in enumerate(text.splitlines(), start=1) if line.strip()]
        if not lines:
            return []
        line_no, last_line = lines[-1]
        digest = hashlib.sha1(last_line.encode("utf-8")).hexdigest()[:10]
        source_key = f"txt:{line_no}:{digest}"
        return [
            SourceCue(
                source_key=source_key,
                source_kind="txt",
                srt_index=None,
                start_ms=None,
                end_ms=None,
                jp_raw=last_line,
            )
        ]


def _parse_srt_ts_ms(raw: str) -> Optional[int]:
    m = SRT_TS_RE.match(raw)
    if not m:
        return None
    hh, mm, ss, mmm = (int(part) for part in m.groups())
    return ((hh * 60 + mm) * 60 + ss) * 1000 + mmm


# Backward compatibility for tools/tests importing SourceReader directly.
SourceReader = FileSourceReader
