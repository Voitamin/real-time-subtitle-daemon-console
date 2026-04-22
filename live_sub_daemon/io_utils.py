from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = 5
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding, dir=str(path.parent), newline="") as tmp:
                tmp.write(text)
                temp_name = tmp.name
            os.replace(temp_name, path)
            return
        except PermissionError as exc:
            last_error = exc
            # Windows often blocks rename when the target is opened without FILE_SHARE_DELETE.
            try:
                with path.open("w", encoding=encoding, newline="") as direct:
                    direct.write(text)
                return
            except Exception as direct_exc:  # noqa: BLE001
                last_error = direct_exc
        except OSError as exc:
            last_error = exc
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

        if attempt < max_attempts - 1:
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error
