from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .models import SourceCue


class BaseSourceReader(ABC):
    @abstractmethod
    def poll(self) -> List[SourceCue]:
        raise NotImplementedError

    def close(self) -> None:
        return None

