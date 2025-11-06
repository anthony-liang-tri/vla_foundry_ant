"""
Rerun Backend Implementation

This file contains the implementation of the RerunBackend class, which provides
logging functionality to rerun.io.
"""

import subprocess
from typing import Any, Mapping

import rerun as rr


class RerunBackend:
    name = "rerun"

    def __init__(self) -> None:
        self._enabled = False
        self._inited = False

    def init(self, run_name: str, *, spawn: bool = True, **kwargs) -> None:
        if not rr.is_enabled():
            self._disable_rerun_analytics()
            try:
                rr.init(run_name, spawn=spawn)
                self._enabled = True
                self._inited = True
            except Exception as e:
                print(f"[rerun_backend] Failed to init rerun: {e}")
                self._enabled = False
                self._inited = True  # Don't keep retrying

    def log(self, path: str, value: Any, **kwargs) -> None:
        if not self._enabled:
            return

        # Pass-through if caller hands us a native rerun object
        try:
            rr.log(path, value, **kwargs)
        except Exception as e:
            print(f"[rerun_backend] rerun.log failed for '{path}': {type(value)} | {e}")

    def flush(self) -> None:
        # rerun flush is implicit; no-op here
        return

    def shutdown(self) -> None:
        # rerun doesn't strictly need it; keep for parity
        return

    def log_dict(
        self,
        data: Mapping[str, Any],
        *,
        base_path: str = "",
        recurse: bool = True,
        sanitize_keys: bool = True,
        max_depth: int = 8,
        _depth: int = 0,
        **kwargs,
    ) -> None:
        if not self._enabled or data is None:
            return
        if _depth > max_depth:
            print("[rerun_backend] log_dict: max_depth exceeded; truncating.")
            return

        def _join(a: str, b: str) -> str:
            if not a:
                return b
            return f"{a.rstrip('/')}/{b.lstrip('/')}"

        def _clean(k: str) -> str:
            if not sanitize_keys:
                return k
            return "".join(ch if ch.isalnum() or ch in "-_./" else "_" for ch in str(k))

        for k, v in data.items():
            key = _clean(k)
            p = _join(base_path, key)

            if recurse and isinstance(v, Mapping):
                self.log_dict(
                    v,
                    base_path=p,
                    recurse=recurse,
                    sanitize_keys=sanitize_keys,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    **kwargs,
                )
                continue

            self.log(p, v, **kwargs)

    def _disable_rerun_analytics(self) -> None:
        try:
            subprocess.run(
                ["rerun", "analytics", "disable"],
                check=True,
                capture_output=True,
                text=True,
            )
            print("[rerun_backend] Rerun analytics disabled.")
        except FileNotFoundError:
            print("[rerun_backend] rerun CLI not found; analytics may be enabled.")
        except subprocess.CalledProcessError as e:
            print(f"[rerun_backend] Failed to disable analytics: {e.stderr or e}")
