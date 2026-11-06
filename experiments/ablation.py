#!/usr/bin/env python3
"""Vary only top_papers; save raw turns, repeated-run checks, metrics, and a plot.

Mock mode measures retrieval and evidence size, never answer quality. Real mode
uses only the explicitly supplied course endpoint. Gold labels are used only for
scoring after chat(), never for retrieval, rewriting, or generation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_dev import hit_rate, load_jsonl
from endpoint_config import resolve_settings
from mock_endpoint import MODEL_NAME, start_mock_endpoint, stop_mock_endpoint
from ta_bot import TABot


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def evaluate(config, single, multi):
    bot = TABot(config)
    rows = []

    def turn(split, item_id, index, session, message, gold):
        captured = []
        format_passages = bot._format_passages

        def capture(papers):
            evidence = format_passages(papers)
            captured.append(len(evidence))
            return evidence

        bot._format_passages = capture
        start = time.perf_counter()
        try:
            result = bot.chat(session, "ablation-user", message)
        finally:
            bot._format_passages = format_passages
        rows.append({"split": split, "id": item_id, "turn": index,
                     "question": message, "gold_paper_ids": gold, **result,
                     "hit_rate": hit_rate(result["citations"], gold),
                     "citation_count": len(result["citations"]),
                     "evidence_chars": sum(captured),
                     "elapsed_seconds": time.perf_counter() - start})

    try:
        for item in single:
            turn("single", item["id"], 0, item["id"], item["question"], item["gold_paper_ids"])
        for item in multi:
            for index, entry in enumerate(item["turns"]):
                turn("multi", item["id"], index, item["id"], entry["content"],
                     entry.get("gold_paper_ids", []))
    finally:
        bot.close()
    return rows


def summarise(rows, multi):
    single_rows = [row for row in rows if row["split"] == "single"]
    multi_rows = [row for row in rows if row["split"] == "multi"]
    last_turns = {item["id"]: len(item["turns"]) - 1 for item in multi}
    final_rows = [row for row in multi_rows if row["turn"] == last_turns[row["id"]]]
    return {"single_hit_rate": mean([row["hit_rate"] for row in single_rows]),
            "multi_final_hit_rate": mean([row["hit_rate"] for row in final_rows]),
            "multi_all_turn_hit_rate": mean([row["hit_rate"] for row in multi_rows]),
            "mean_citations": mean([row["citation_count"] for row in rows]),
            "mean_evidence_chars": mean([row["evidence_chars"] for row in rows]),
            "mean_elapsed_seconds": mean([row["elapsed_seconds"] for row in rows])}


def plot_results(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["settings"]
    x = [row["top_papers"] for row in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 2.8))
    for key, label, marker, linestyle in (
        ("single_hit_rate", "Single turn (10)", "o", "-"),
        ("multi_final_hit_rate", "Dialogue endings (3)", "s", "--"),
        ("multi_all_turn_hit_rate", "All dialogue turns (10)", "^", ":"),
    ):
        left.plot(x, [100 * row["metrics"][key] for row in rows],
                  marker=marker, linestyle=linestyle, label=label, alpha=0.85)
    left.set(xlabel="Papers per query (k)", ylabel="Gold paper recall (%)", ylim=(0, 108), xticks=x)
    left.legend(loc="lower right", fontsize=8)
    right.plot(x, [row["metrics"]["mean_evidence_chars"] / 1000 for row in rows],
               "o-", color="#226699")
    right.set(xlabel="Papers per query (k)", ylabel="Mean evidence (1,000 chars)", xticks=x)
    for axis in (left, right):
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    mode = result["metadata"]["mode"]
    fig.suptitle("Retrieval depth ablation · " + mode + " endpoint · no answer-quality judge", fontsize=10)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(str(directory / ("ablation." + suffix)), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true")
    group.add_argument("--endpoint", help="Course vLLM endpoint, including an optional /v1 suffix")
    parser.add_argument("--model", help="Served model name; auto-detected if omitted")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    parser.add_argument("--output", type=Path, help="Default: results for mock, results-course for course")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("results" if args.mock else "results-course")
    if args.repeats < 2 or any(k < 1 for k in args.k):
        parser.error("Use at least two repeats and positive k values")
    single = load_jsonl(ROOT / "dev_eval/single_turn.jsonl")
    multi = load_jsonl(ROOT / "dev_eval/multi_turn.jsonl")
    if not single or not multi:
        parser.error("Both public retrieval datasets must be present")
    server = None
    if args.mock:
        endpoint, server = start_mock_endpoint()
        model = args.model or MODEL_NAME
    else:
        try:
            endpoint, model = resolve_settings(args.endpoint, args.model, args.timeout)
        except (ValueError, TimeoutError) as exc:
            parser.exit(1, str(exc) + "\n")
    config = dict(endpoint=endpoint, model=model, corpus_path=str(args.corpus),
                  temperature=0, seed=args.seed, timeout=args.timeout, rewrite=True)
    files = [ROOT / "ta_bot.py", ROOT / "mock_endpoint.py", ROOT / "endpoint_config.py",
             ROOT / "experiments/ablation.py",
             args.corpus / "corpus.db", ROOT / "dev_eval/single_turn.jsonl",
             ROOT / "dev_eval/multi_turn.jsonl"]
    metadata = {"mode": "mock" if args.mock else "course", "model": config["model"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                "requests": importlib.metadata.version("requests"),
                "platform": platform.platform(), "seed": args.seed, "temperature": 0,
                "rewrite": True, "repeats": args.repeats,
                "single_questions": len(single), "dialogues": len(multi),
                "dialogue_turns": sum(len(item["turns"]) for item in multi),
                "answer_quality": None,
                "sha256": {path.name: sha256(path) for path in files},
                "limitations": "Small public dev set; mock completions cannot establish answer quality. "
                "Citations identify supplied evidence; contextual carry-over can exceed k. "
                "Wall time includes endpoint overhead and is not a graded-server estimate."}
    result = {"metadata": metadata, "settings": []}
    try:
        for k in sorted(set(args.k)):
            runs = [evaluate(dict(config, top_papers=k), single, multi)
                    for _ in range(args.repeats)]
            signature = lambda rows: [(r["citations"], r["answer"], r["evidence_chars"]) for r in rows]
            repeat_match = all(signature(run) == signature(runs[0]) for run in runs[1:])
            metrics = summarise(runs[0], multi)
            result["settings"].append({"top_papers": k, "metrics": metrics,
                                        "identical_repeats": repeat_match, "runs": runs})
            print("k={}: single={:.1%}, dialogue final={:.1%}, all dialogue turns={:.1%}, "
                  "evidence={:.0f} chars, identical={}".format(
                      k, metrics["single_hit_rate"], metrics["multi_final_hit_rate"],
                      metrics["multi_all_turn_hit_rate"], metrics["mean_evidence_chars"], repeat_match))
    finally:
        if server:
            stop_mock_endpoint(server)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ablation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_results(result, args.output)
    print("Saved raw measurements and plots to", args.output)
    if not all(setting["identical_repeats"] for setting in result["settings"]):
        raise SystemExit("Repeated outputs differed; inspect the recorded runs")


if __name__ == "__main__":
    main()
