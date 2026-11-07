#!/usr/bin/env python3
"""Verify the saved local study and document the observed LoRA scoring error."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "report/bonus/local_judge_run"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    results = json.loads((RUN / "results.json").read_text())
    protocol = results["protocol"]
    rows = results["rows"]
    assert json.loads((RUN / "completed.json").read_text())["answers_and_judgments"] == len(rows) == 20
    assert len({row["id"] for row in rows}) == 20
    for path, digest in protocol["sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for row in rows:
        content = row["bot_model_calls"][-1]["response"]["choices"][0]["message"]["content"]
        assert content.strip() == row["answer"]
        assert row["answer"] in row["judge_call"]["request"]["messages"][-1]["content"]
        assert row["evidence_text"] in row["judge_call"]["request"]["messages"][-1]["content"]
    singles = [row for row in rows if row["id"].startswith("dev-st-")]
    final_ids = ("dev-mt-01/turn-3", "dev-mt-02/turn-4", "dev-mt-03/turn-3")
    finals = [row for row in rows if row["id"] in final_ids]
    recall = lambda row: len(set(row["gold_paper_ids"]) & set(row["citations"])) / len(set(row["gold_paper_ids"]))
    memory_log = (RUN / "memory_probes.txt").read_text()
    assert "test_all_probes" in memory_log and memory_log.rstrip().endswith("OK")
    summary = {"generator_and_local_judge": protocol["generator_model"],
               "public_answers": len(rows), "single_questions": len(singles),
               "single_hit_rate": sum(map(recall, singles)) / len(singles),
               "dialogue_endings": len(finals), "multi_final_hit_rate": sum(map(recall, finals)) / len(finals),
               "valid_json_judgments": sum(row["judge"] is not None for row in rows),
               "invalid_json_judgments": [row["id"] for row in rows if row["judge"] is None],
               "memory_probes_passed": 12, "official_course_judge": False,
               "answer_quality_threshold_pass": None,
               "note": "No calibrated quality score is inferred from this local self-judge."}
    write_json(RUN / "summary.json", summary)

    case = next(row for row in rows if row["id"] == "dev-st-06/turn-1")
    scores = ("accuracy", "relevance", "completeness", "grounding")
    assert all(case["judge"][key] == 5 for key in scores)
    assert case["judge"]["unsupported_claims"] == []
    repeats = json.loads((RUN / "lora_judge_repeats.json").read_text())
    assert len(repeats["repeats"]) == 2
    for repeat in repeats["repeats"]:
        assert repeat["request"] == case["judge_call"]["request"]
        score = json.loads(repeat["response"]["choices"][0]["message"]["content"])
        assert all(score[key] == 5 for key in scores) and score["unsupported_claims"] == []
    wrong_claim = case["answer"].split("This is in contrast to full fine-tuning,", 1)[1]
    wrong_claim = "This is in contrast to full fine-tuning," + wrong_claim
    connection = sqlite3.connect(str(ROOT / "corpus/corpus.db"))
    paper_metadata = connection.execute(
        "SELECT title, arxiv_id, url_abs FROM papers WHERE paper_id = ?", ("L11-01",)).fetchone()
    sources = []
    for paper in case["evidence_papers"]:
        if paper["paper_id"] != "L11-01":
            continue
        for chunk in paper["chunks"]:
            match = connection.execute(
                "SELECT chunk_id, section FROM chunks WHERE paper_id = ? AND text = ?",
                (paper["paper_id"], chunk["text"])).fetchone()
            assert match
            sources.append({"paper_id": paper["paper_id"], "chunk_id": match[0], "section": match[1],
                            "text": chunk["text"], "sha256": hashlib.sha256(chunk["text"].encode()).hexdigest()})
    connection.close()
    assert any("which retrains all model parameters" in source["text"] for source in sources)
    assert any("number of adapter layers" in source["text"] for source in sources)
    record = {"status": "observed_local_judge_error_with_raw_evidence",
              "official_course_judge": False,
              "bonus_eligibility": "For staff to determine; a local self-judge case, not a claim about the official grader.",
              "selection": "One concrete factual counterexample selected from all 20 public dev turns; the complete run is retained.",
              "case_id": case["id"], "question": case["question"], "unedited_bot_answer": case["answer"],
              "incorrect_claim_verbatim": wrong_claim,
              "model": protocol["generator_model"], "temperature": 0, "seed": 42,
              "judge": case["judge"], "original_judge_call": case["judge_call"],
              "source_evidence_supplied_to_judge": sources,
              "paper": {"paper_id": "L11-01", "title": paper_metadata[0],
                        "arxiv_id": paper_metadata[1], "url": paper_metadata[2]},
              "factual_correction": "The expression containing L_Adpt and L_LN counts adapter-layer and trainable LayerNorm parameters. It is not the full-fine-tuning parameter count. Full fine-tuning retrains all original model parameters, as the supplied abstract states.",
              "why_score_is_wrong": "The answer explicitly attributes an adapter formula to full fine-tuning. The evidence supplied to the judge contradicts that attribution, yet the judge assigns 5/5 accuracy and grounding and identifies no unsupported claims.",
              "repeated_identical_requests": 3,
              "repeat_outcome": "All four criteria remained 5/5, with an empty unsupported_claims list, on all three requests.",
              "limits": "This is a same-model local self-judge on one checkpoint and one case. It does not establish the official course judge's behavior, a population error rate, or an exact corrected numeric grade.",
              "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                         (RUN / "protocol.json", RUN / "results.json", RUN / "lora_judge_repeats.json",
                          ROOT / "local_inference/server_manifest.json")}}
    write_json(ROOT / "report/bonus/local_lora_case.json", record)
    text = f"""Observed local judge error: adapter parameters mislabelled as full fine-tuning

Status and scope

This is a newly executed local development experiment, not the official course
judge. Both generator and judge are meta-llama/Llama-3.1-8B-Instruct, served with
vLLM on one NVIDIA A100 80GB GPU, bfloat16, temperature 0, seed 42. The model
checkpoint revision and server command are in local_inference/server_manifest.json.
Staff determine whether this clearly labelled local case earns the assignment bonus.

Question: {case['question']}
Public case: {case['id']}

Actual bot claim (verbatim; part of the unedited saved answer)

{wrong_claim}

Actual local judge output

{json.dumps(case['judge'],indent=2,ensure_ascii=False)}

Why this judgment is wrong

The LoRA passage explicitly identifies L_Adpt as the number of adapter layers
and L_LN as the number of trainable LayerNorms. The expression therefore counts
adapter-based tuning parameters. The same supplied paper's abstract defines full
fine-tuning as retraining all model parameters. The answer mislabels the adapter
expression as full fine-tuning: this is a concrete factual attribution error in
the key parameter-count comparison, not a preference about writing style.

These passages were actually included in the judge's request. Despite having the
contradicting evidence, it gives full accuracy and grounding marks and reports
no unsupported claims. Replaying the identical original judge request twice more
gives the same four 5/5 scores and empty unsupported_claims list: three of three.

Corrected explanation

LoRA freezes the original weights and learns low-rank factors. Adapter tuning
learns added adapter-layer parameters and optionally LayerNorm parameters. Full
fine-tuning retrains the original model parameters. The answer's adapter formula
must be attributed to adapters, not to full fine-tuning.

Evidence and verification

Full unedited answer, original judge request/response, source chunks, exact
incorrect excerpt, repeats, model identity and checksums: local_lora_case.json.
All twenty public answers and all twenty judge replies: local_judge_run/.
The judge prompt was fixed before answers were generated; no hidden judge prompt
or hidden test data was inspected. Answers were saved before they were judged.
Two of twenty judge replies were malformed JSON and remain preserved unchanged;
this case's original and repeated judgments are valid JSON. No overall calibrated
answer-quality score or exact replacement grade is claimed.

Limits

This demonstrates an observed local self-judge failure, not an error by the
official course grader or a general failure rate. Same-model self-judging is a
limitation; this single case does not isolate the cause of the error. The earlier
tau-bench anecdote remains a separate historical candidate, not this new run.
"""
    (ROOT / "report/bonus/LOCAL_JUDGE_CASE.txt").write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Verified and documented local LoRA judge error; all 3 identical requests scored 5/5.")


if __name__ == "__main__":
    main()
