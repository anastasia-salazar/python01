"""Logging: records every prompt, raw output, error and check to a JSON Lines file."""

import json
from datetime import datetime, timezone
from pathlib import Path


class InteractionLogger:
    """Writes one JSON object per line, so the log is easy to read and to process."""

    def __init__(self, log_dir):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"discussion_{timestamp}.jsonl"

    def log(self, event, **details):
        entry = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
