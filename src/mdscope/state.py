from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, outdir: Path) -> None:
        self.outdir = outdir
        self.state_dir = outdir / ".state"
        self.logs_dir = outdir / "logs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def is_done(self, step: str) -> bool:
        return (self.state_dir / f"{step}.done").exists()

    def mark_done(self, step: str, metadata: dict[str, Any]) -> None:
        (self.state_dir / f"{step}.done").write_text("done\n")
        (self.state_dir / f"{step}.meta.json").write_text(json.dumps(metadata, indent=2))

    def clear_done(self, step: str) -> None:
        for suffix in ("done", "meta.json"):
            path = self.state_dir / f"{step}.{suffix}"
            if path.exists():
                path.unlink()

    def log_error(self, step: str, error: str) -> None:
        error_path = self.logs_dir / "error.json"
        payload = []
        if error_path.exists():
            try:
                payload = json.loads(error_path.read_text())
            except Exception:
                payload = []
        payload.append({"step": step, "error": error})
        error_path.write_text(json.dumps(payload, indent=2))


def short_config_hash(config_text: str) -> str:
    digest = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    return digest[:12]
