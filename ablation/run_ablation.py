#!/usr/bin/env python3
"""Ablation study: does rewriting follow-up questions improve retrieval?

Each dialogue is replayed with rewriting off and on, and only the last turn is
scored. Follow-ups lean on the conversation; controls are standalone topic
switches, where a rewrite can only hurt. Tune on the "tune" split and report the
"heldout" split, so the reported numbers are not fitted to the questions.

    python ablation/run_ablation.py --mock
    python ablation/run_ablation.py --endpoint URL --model NAME --split heldout
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mock_endpoint import start_mock_endpoint, stop_mock_endpoint  # noqa: E402
from ta_bot import TABot  # noqa: E402

CONDITIONS = {"rewrite_off": False, "rewrite_on": True}
BOOTSTRAP_SAMPLES = 2000


def replay(bot: TABot, dialogue: dict) -> dict:
    """Run one dialogue; an endpoint failure counts as a miss but is reported."""
    try:
        for turn in dialogue["turns"]:
            result = bot.chat(dialogue["id"], "ablation_user", turn)
    except TimeoutError as exc:
        return {"hit": False, "citations": [], "error": str(exc)}
    return {
        "hit": dialogue["gold_paper_id"] in result["citations"],
        "citations": result["citations"],
        "error": None,
    }


def bootstrap_ci(hits: list[bool], seed: int) -> tuple[float, float]:
    """95% percentile interval for the hit-rate."""
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(BOOTSTRAP_SAMPLES)
    )
    return means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


def mcnemar_exact(off: list[bool], on: list[bool]) -> tuple[int, int, float]:
    """Two-sided exact McNemar test: the paired test, as both runs share questions."""
    only_on = sum(1 for a, b in zip(off, on) if b and not a)
    only_off = sum(1 for a, b in zip(off, on) if a and not b)
    discordant = only_on + only_off
    if discordant == 0:
        return only_on, only_off, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_on, only_off) + 1))
    return only_on, only_off, min(1.0, 2 * tail / 2**discordant)


def summarize(rows: list[dict], seed: int) -> dict:
    """Hit-rate per condition with an interval, plus the paired comparison."""
    summary: dict = {"n": len(rows)}
    for condition in CONDITIONS:
        hits = [row[condition]["hit"] for row in rows]
        low, high = bootstrap_ci(hits, seed)
        summary[condition] = {"hit_rate": sum(hits) / len(hits), "ci_low": low, "ci_high": high}
    only_on, only_off, p_value = mcnemar_exact(
        [row["rewrite_off"]["hit"] for row in rows], [row["rewrite_on"]["hit"] for row in rows]
    )
    summary["paired"] = {"only_rewrite_on": only_on, "only_rewrite_off": only_off, "p": p_value}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mock", action="store_true", help="smoke-test the wiring only")
    parser.add_argument("--endpoint")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--corpus", default=str(ROOT / "corpus"))
    parser.add_argument("--split", choices=("tune", "heldout", "all"), default="tune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "ablation" / "results.json"))
    args = parser.parse_args()
    if bool(args.mock) == bool(args.endpoint):
        parser.error("give exactly one of --mock or --endpoint")

    server = None
    endpoint = args.endpoint
    if args.mock:
        endpoint, server = start_mock_endpoint()

    with open(ROOT / "ablation" / "dialogues.jsonl") as f:
        dialogues = [json.loads(line) for line in f if line.strip()]
    if args.split != "all":
        dialogues = [d for d in dialogues if d["split"] == args.split]

    rows = []
    for index, dialogue in enumerate(dialogues, 1):
        row = {key: dialogue[key] for key in ("id", "kind", "gold_paper_id")}
        for condition, rewrite in CONDITIONS.items():
            # A fresh bot per run stops one condition's state reaching the other.
            bot = TABot(
                {
                    "endpoint": endpoint,
                    "model": args.model,
                    "corpus_path": args.corpus,
                    "temperature": 0.0,
                    "seed": args.seed,
                    "rewrite": rewrite,
                }
            )
            row[condition] = replay(bot, dialogue)
        rows.append(row)
        print(
            f"[{index}/{len(dialogues)}] {row['id']}: "
            f"off={'hit' if row['rewrite_off']['hit'] else 'miss'} "
            f"on={'hit' if row['rewrite_on']['hit'] else 'miss'}",
            flush=True,
        )

    if server:
        stop_mock_endpoint(server)

    results = {
        "model": args.model,
        "mock": args.mock,
        "split": args.split,
        "seed": args.seed,
        "failed_runs": sum(1 for r in rows for c in CONDITIONS if r[c]["error"]),
        "summary": {
            kind: summarize([r for r in rows if r["kind"] == kind], args.seed)
            for kind in ("followup", "control")
            if any(r["kind"] == kind for r in rows)
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("failed_runs", "summary")}, indent=2))


if __name__ == "__main__":
    main()
