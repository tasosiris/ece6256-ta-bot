#!/usr/bin/env python3
"""Run offline regressions and dev probes; save reviewable logs and a JSON record."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    print(log, end="")
    (output / "unit_tests.txt").write_text(log, encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(ROOT / "eval_dev.py"), "--mock", "--extra-memory-probes",
         str(ROOT / "tests/memory_probes_extra.jsonl")],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    print(run.stdout, end="")
    (output / "dev_eval.txt").write_text(run.stdout, encoding="utf-8")
    single = re.search(r"Single-turn hit-rate:\s+([\d.]+)%", run.stdout)
    multi = re.search(r"Multi-turn hit-rate:\s+([\d.]+)%", run.stdout)
    memory = re.search(r"Memory probes:\s+(\d+)/(\d+) passed", run.stdout)
    evaluation_ok = bool(run.returncode == 0 and single and multi and memory
                         and float(single.group(1)) >= 70 and float(multi.group(1)) >= 70
                         and memory.group(1) == memory.group(2))
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "mode": "mock", "answer_quality": None,
              "unit_tests": {"run": test_result.testsRun, "passed": test_result.wasSuccessful(),
                             "failures": len(test_result.failures), "errors": len(test_result.errors)},
              "dev_eval": {"passed": evaluation_ok,
                           "single_hit_rate": float(single.group(1)) / 100 if single else None,
                           "multi_final_hit_rate": float(multi.group(1)) / 100 if multi else None,
                           "memory_passed": int(memory.group(1)) if memory else None,
                           "memory_total": int(memory.group(2)) if memory else None,
                           "public_memory_probes": 2, "additional_positive": 2,
                           "additional_negative": 8},
              "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in ("ta_bot.py", "mock_endpoint.py", "endpoint_config.py", "eval_dev.py",
                                      "tests/test_ta_bot.py", "tests/memory_probes_extra.jsonl")}}
    (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not test_result.wasSuccessful() or not evaluation_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
