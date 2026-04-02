from __future__ import annotations

import json
from pathlib import Path

from .batch_models import BatchDiscoveryResult


def save_batch_discovery(result: BatchDiscoveryResult, path: str) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2), encoding='utf-8')


def load_batch_discovery(path: str) -> BatchDiscoveryResult:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return BatchDiscoveryResult.from_dict(data)
