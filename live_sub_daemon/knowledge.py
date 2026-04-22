from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class KnowledgeSnapshot:
    glossary: Dict[str, str]
    names_whitelist: List[str]


class KnowledgeStore:
    def __init__(self, glossary_path: Path, names_path: Path, reload_interval_sec: float):
        self.glossary_path = glossary_path
        self.names_path = names_path
        self.reload_interval_sec = reload_interval_sec
        self._lock = threading.Lock()
        self._last_reload_mono = 0.0
        self._last_mtime: Tuple[float, float] = (0.0, 0.0)
        self._snapshot = KnowledgeSnapshot(glossary={}, names_whitelist=[])
        self._reload(force=True)

    def get_snapshot(self) -> KnowledgeSnapshot:
        self._reload(force=False)
        with self._lock:
            return KnowledgeSnapshot(
                glossary=dict(self._snapshot.glossary),
                names_whitelist=list(self._snapshot.names_whitelist),
            )

    def _reload(self, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last_reload_mono < self.reload_interval_sec:
            return
        self._last_reload_mono = now

        glossary_mtime = self.glossary_path.stat().st_mtime if self.glossary_path.exists() else 0.0
        names_mtime = self.names_path.stat().st_mtime if self.names_path.exists() else 0.0
        current = (glossary_mtime, names_mtime)

        if not force and current == self._last_mtime:
            return

        glossary = _load_glossary(self.glossary_path)
        names = _load_names(self.names_path)
        with self._lock:
            self._snapshot = KnowledgeSnapshot(glossary=glossary, names_whitelist=names)
            self._last_mtime = current


def _load_glossary(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            ja, zh = line.split("\t", 1)
        elif "," in line:
            ja, zh = line.split(",", 1)
        else:
            continue
        ja = ja.strip()
        zh = zh.strip()
        if ja and zh:
            mapping[ja] = zh
    return mapping


def _load_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    names: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        content = line
        if "\t" in line:
            content, _ = line.split("\t", 1)
        elif "|" in line:
            content, _ = line.split("|", 1)
        name = content.strip()
        if not name:
            continue
        names.append(name)
    return names
