#!/usr/bin/env python3
"""Validate and save the staff-provided course URL and its served model locally."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from endpoint_config import LOCAL_CONFIG, list_models, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Course vLLM URL from the staff")
    parser.add_argument("--model", help="Optional when the endpoint serves exactly one model")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    try:
        endpoint, model = load_settings(args.endpoint, args.model)
        models = list_models(endpoint, args.timeout)
        if model and model not in models:
            raise ValueError("The requested model is not served; choose from: " + ", ".join(models))
        if not model:
            if len(models) != 1:
                raise ValueError("Select --model from: " + ", ".join(models))
            model = models[0]
    except (ValueError, TimeoutError) as exc:
        parser.exit(1, str(exc) + "\n")
    # Write only after the server has confirmed the model; this file is gitignored.
    with open(str(LOCAL_CONFIG), "w", encoding="utf-8") as stream:
        json.dump({"endpoint": endpoint, "model": model}, stream, indent=2)
        stream.write("\n")
    os.chmod(str(LOCAL_CONFIG), 0o600)
    print("Saved verified course endpoint settings to", LOCAL_CONFIG.name)
    print("Served model:", model)
    print("Run: python eval_dev.py")
    print("Ablation: python experiments/ablation.py --output results-course")


if __name__ == "__main__":
    main()
