#!/usr/bin/env python3
"""Rebuild the revised three-page report from archived experiment evidence.

    python -m pip install -r requirements.txt -r report/requirements.txt
    python report/build_report.py --authors "Full name and Full name"

Saved real-model results are never overwritten. The mock replay and local test
logs in report/validation document checks performed for this revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                               SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
rl_config.useA85 = 0  # Binary compressed streams; keeps PDF changes out of text diffs.


def wilson(hits, n):
    z = 1.959963984540054
    p = hits / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def analyze():
    real = json.loads((ROOT / "ablation/results.json").read_text())
    mock = json.loads((HERE / "validation/mock_heldout.json").read_text())
    dialogues = [json.loads(line) for line in (ROOT / "ablation/dialogues.jsonl").read_text().splitlines()
                 if line.strip()]
    assert real["mock"] is False and mock["mock"] is True
    assert real["split"] == mock["split"] == "heldout"
    assert real["seed"] == mock["seed"] == 42
    by_id = {item["id"]: item for item in dialogues if item["split"] == "heldout"}
    assert len(by_id) == len(real["rows"]) == len(mock["rows"]) == 20
    assert {row["id"] for row in real["rows"]} == {row["id"] for row in mock["rows"]} == set(by_id)
    mock_by_id = {row["id"]: row for row in mock["rows"]}
    result = {"model": real["model"], "seed": real["seed"],
              "split_counts": {f"{split}/{kind}": n for (split, kind), n in
                               Counter((d["split"], d["kind"]) for d in dialogues).items()},
              "groups": {}, "recoveries": [], "misses": []}
    for condition in ("rewrite_off", "rewrite_on"):
        for row in real["rows"] + mock["rows"]:
            assert row[condition]["hit"] == (row["gold_paper_id"] in row[condition]["citations"])
            assert row[condition]["error"] is None
    for row in real["rows"]:
        assert row["gold_paper_id"] == by_id[row["id"]]["gold_paper_id"]
        assert row["kind"] == by_id[row["id"]]["kind"]
        off, on = row["rewrite_off"]["hit"], row["rewrite_on"]["hit"]
        if on and not off:
            gold = row["gold_paper_id"]
            result["recoveries"].append({"id": row["id"], "paper_id": gold,
                                         "real_rank": row["rewrite_on"]["citations"].index(gold) + 1,
                                         "mock_rank": mock_by_id[row["id"]]["rewrite_on"]["citations"].index(gold) + 1})
        if not on:
            result["misses"].append({"id": row["id"], "turns": by_id[row["id"]]["turns"]})
    for kind in ("followup", "control"):
        rows = [row for row in real["rows"] if row["kind"] == kind]
        n = len(rows)
        group = {"n": n}
        for condition in ("rewrite_off", "rewrite_on"):
            hits = sum(row[condition]["hit"] for row in rows)
            rate = hits / n
            assert rate == real["summary"][kind][condition]["hit_rate"]
            group[condition] = {"hits": hits, "rate": rate, "wilson95": wilson(hits, n)}
        gains = sum(row["rewrite_on"]["hit"] and not row["rewrite_off"]["hit"] for row in rows)
        losses = sum(row["rewrite_off"]["hit"] and not row["rewrite_on"]["hit"] for row in rows)
        discordant = gains + losses
        p = (min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
                 / 2 ** discordant) if discordant else 1.0)
        paired = real["summary"][kind]["paired"]
        assert (gains, losses, p) == (paired["only_rewrite_on"], paired["only_rewrite_off"], paired["p"])
        differences = [int(row["rewrite_on"]["hit"]) - int(row["rewrite_off"]["hit"]) for row in rows]
        rng = random.Random(real["seed"])
        boot = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(2000))
        group["paired"] = {"gains": gains, "losses": losses, "p": p,
                           "difference": sum(differences) / n,
                           "bootstrap95": [boot[50], boot[1949]]}
        result["groups"][kind] = group
    result["off_citations_match_mock"] = all(
        row["rewrite_off"]["citations"] == mock_by_id[row["id"]]["rewrite_off"]["citations"]
        for row in real["rows"])
    assert result["off_citations_match_mock"]
    assert all(item["real_rank"] <= 3 and item["mock_rank"] == 4 for item in result["recoveries"])
    unit_log = (HERE / "validation/unit_tests.txt").read_text()
    match = re.search(r"Ran (\d+) tests", unit_log)
    assert match and re.search(r"\nOK\s*$", unit_log)
    result["offline_tests"] = int(match.group(1))
    dev_log = (HERE / "validation/dev_eval_mock.txt").read_text()
    for expected in ("Single-turn hit-rate:  100.0%", "Multi-turn hit-rate:   100.0%", "2/2 passed"):
        assert expected in dev_log
    result["runtime"] = {"python": platform.python_version(), "matplotlib": matplotlib.__version__}
    local_case = json.loads((HERE / "bonus/local_lora_case.json").read_text())
    local_summary = json.loads((HERE / "bonus/local_judge_run/summary.json").read_text())
    assert local_case["status"] == "observed_local_judge_error_with_raw_evidence"
    assert local_case["official_course_judge"] is False
    assert local_case["repeated_identical_requests"] == 3
    assert all(local_case["judge"][key] == 5 for key in ("accuracy", "relevance", "completeness", "grounding"))
    for path, digest in local_case["sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    result["local_judge_case"] = {"id": local_case["case_id"], "model": local_case["model"],
                                 "scores": local_case["judge"], "repeat_count": 3,
                                 "official_course_judge": False, "study_summary": local_summary}
    result["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    files = [ROOT / "ta_bot.py", ROOT / "ablation/results.json", ROOT / "ablation/dialogues.jsonl",
             ROOT / "ablation/run_ablation.py", ROOT / "corpus/corpus.db",
             ROOT / "tests/test_bot.py", ROOT / "tests/test_grading.py", ROOT / "tests/test_ta_bot.py",
             HERE / "build_report.py",
             HERE / "validation/mock_heldout.json", HERE / "validation/unit_tests.txt",
             HERE / "validation/dev_eval_mock.txt", HERE / "bonus/local_lora_case.json",
             HERE / "bonus/local_judge_run/summary.json"]
    result["sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in files}
    result["provenance"] = {
        "real_ablation": "Archived collaborator rewriting ablation; no new run of that ablation.",
        "hardware_and_prior_dev_scores": "Reported in the collaborator draft at commit 3a7353c; raw logs unavailable.",
        "mock_and_offline_tests": "Re-executed for this revision; logs and mock outcomes archived here.",
        "answer_quality": "A new local evidence-aware self-judge run and a reproduced LoRA scoring error are archived. Official course scores and calibrated quality remain unmeasured."}
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_plot(data):
    fig, ax = plt.subplots(figsize=(6.8, 2.1), dpi=220)
    palette = ("#2166ac", "#ce5a23")
    for index, condition in enumerate(("rewrite_off", "rewrite_on")):
        xs = [x + (index - 0.5) * 0.32 for x in range(2)]
        groups = [data["groups"][kind] for kind in ("followup", "control")]
        rates = [100 * group[condition]["rate"] for group in groups]
        errors = [[rate - 100 * group[condition]["wilson95"][0] for rate, group in zip(rates, groups)],
                  [100 * group[condition]["wilson95"][1] - rate for rate, group in zip(rates, groups)]]
        ax.bar(xs, rates, width=0.30, color=palette[index], label="Off" if index == 0 else "On",
               yerr=errors, capsize=3, error_kw={"elinewidth": 1}, zorder=3)
        for x, rate, group in zip(xs, rates, groups):
            ax.text(x, 4, f'{group[condition]["hits"]}/{group["n"]}', ha="center", va="bottom",
                    color="white", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Follow-ups (n=15)", "Topic-switch controls (n=5)"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Gold paper retrieved (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=9)
    ax.legend(title="rewrite flag", ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.30),
              frameon=False, fontsize=9, title_fontsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.18, zorder=0)
    fig.tight_layout(pad=0.5)
    for extension in ("png", "pdf"):
        fig.savefig(str(HERE / ("ablation_v2." + extension)), bbox_inches="tight")
    plt.close(fig)


def build(data, authors):
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5, leading=12.2, spaceAfter=6)
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=18, leading=22,
                           textColor=colors.HexColor("#163b55"), spaceAfter=7)
    heading_style = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=11,
                                   leading=14, spaceBefore=8, spaceAfter=5,
                                   textColor=colors.HexColor("#163b55"))
    small = ParagraphStyle("Small", fontName="Helvetica", fontSize=8, leading=10, spaceAfter=6)
    code = ParagraphStyle("Code", fontName="Courier", fontSize=7.3, leading=9, spaceAfter=5)
    story = []

    def p(text):
        story.append(Paragraph(text, body))

    def heading(text):
        story.append(Paragraph(text, heading_style))

    def table(rows, widths):
        cells = [[Paragraph(str(cell), small) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.HexColor("#92a7b6")),
            ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.HexColor("#bdcbd4")),
        ]))
        story.extend([item, Spacer(1, 5)])

    story.append(Paragraph("The Course TA-Bot: Retrieval and Memory", title))
    subtitle = "ECE 6256 / CS 6604 · Assignment 3 · Revised November 6, 2026"
    if authors:
        subtitle += "<br/>" + escape(authors)
    story.append(Paragraph(subtitle, small))
    heading("1. System and real-model experiment")
    p("The bot retrieves from 108 papers and 5,711 supplied chunks using SQLite FTS5 BM25. "
      "Title terms receive weight 5. From 40 candidate chunks, paper ranks sum the best two "
      "scores; up to three passages are retained for each of the top three papers. Original "
      "and rewritten searches are combined by reciprocal rank fusion (damping 60). Generation "
      "receives attributed evidence, up to three previous exchanges, and a filtered student "
      "profile. The prompt requests 100–250 words, paper titles and IDs, and explicit abstention "
      "when evidence is incomplete; the answer budget is 800 tokens. Returned citation IDs "
      "come from supplied retrieval evidence, not from parsing model text.")
    p("The archived experiment uses <b>meta-llama/Llama-3.1-8B-Instruct</b>, temperature 0 and "
      "seed 42. The prior team draft describes serving it with vLLM on a PACE-ICE A40, rather than "
      "the staff's shared endpoint. Model/seed/results are saved; hardware details and earlier "
      "public-dev logs survive only as descriptions in that draft. This revision also runs "
      "the cached 8B checkpoint locally on an A100 80GB (bfloat16, vLLM 0.19.1, 16K context) "
      "for a new public-dev and self-judge study. Both runs are separate from the official grader.")
    heading("2. Ablation: the rewrite flag (R4)")
    p("<b>Design.</b> Forty authored two-turn dialogues are split into 20 tuning and 20 labelled "
      "held-out cases. The reported split contains 15 context-dependent follow-ups and five "
      "standalone topic switches. Only <b>rewrite</b> changes; corpus, retrieval parameters, "
      "prompts, model and seed are fixed. Each condition starts a fresh bot. A hit means the "
      "final turn's citations contain the specified gold paper. The 20 cases yield 40 "
      "condition-dialogue runs, with no recorded endpoint failures. The flag activates both "
      "query rewriting and last-subject fallback; it is not a pure rewriting intervention.")
    story.append(Image(str(HERE / "ablation_v2.png"), width=7.15 * inch, height=2.05 * inch))
    story.append(Paragraph("Figure 1. Archived real-model outcomes; error bars are 95% Wilson intervals "
                           "recomputed from the saved counts. Unlike the original bootstrap plot, "
                           "these retain uncertainty for five successes out of five. On/off observations "
                           "are paired by dialogue; significance is assessed with exact McNemar, not "
                           "overlap of marginal intervals.", small))
    table([["Group", "Flag off", "Flag on", "Paired gains / losses", "Exact p"],
           ["Follow-ups", "6/15 (40.0%)", "14/15 (93.3%)", "8 / 0", "0.0078125"],
           ["Topic-switch controls", "5/5 (100%)", "5/5 (100%)", "0 / 0", "1.0"]],
          [132, 90, 90, 130, 72])
    diff = data["groups"]["followup"]["paired"]
    p("<b>Result.</b> Follow-up recall improves by <b>53.3 percentage points</b> (paired "
      "percentile-bootstrap 95% interval {:.1f}–{:.1f} points; 2,000 resamples, seed 42). "
      "The controls show no observed regression, but 5/5 has a Wilson interval of 56.6–100%; "
      "it cannot establish safety on arbitrary topic switches.".format(
          100 * diff["bootstrap95"][0], 100 * diff["bootstrap95"][1]))
    story.append(PageBreak())

    heading("3. What produced the gain?")
    p("<b>Two sufficient routes on these examples.</b> In the real-model records, all eight "
      "newly recovered gold papers occur at rank 2 or 3. The code appends last-subject "
      "fallback only after the fused top three, so these recoveries entered through fused "
      "retrieval, rather than through the append step. A new replay of the same split with "
      "the canned mock also gives 6/15 versus 14/15, with the same eight recoveries all at "
      "rank 4. The flag-off citation lists match the archived real run exactly. Thus fallback "
      "alone is sufficient for the observed recall gain, while real rewriting recovers those "
      "papers within the top three. This diagnostic does not establish an additional recall "
      "benefit of rewriting over fallback, or any answer-quality advantage from a higher rank.")
    p("<b>Failure and validity.</b> The remaining miss, followup-18, asks 'What is BERT?' then "
      "'Which datasets showed the largest gains?' The second question has no recognized "
      "back-reference, so the contextual branch is skipped. The constructed follow-ups all "
      "concern the first turn's gold paper, favoring subject carry-over. The prior team draft "
      "also reports replaying the labelled held-out split during development: it is not a "
      "pristine test set. The sample is small, with one model and one seed. These results "
      "support contextual retrieval on this development set, not a general claim that query "
      "rewriting alone adds 53.3 points or meets the course's answer-quality threshold.")
    heading("4. Memory design and privacy (R3)")
    p("<b>Store and update.</b> SQLite facts are keyed by user_id and contain one of five "
      "allowlisted categories: name, project, interest, academic background, and response "
      "preference. Explicit user statements trigger deterministic extraction before retrieval; "
      "assistant output and paper text never create facts. The latest name/project replaces "
      "the previous value. Other categories can coexist, duplicate facts are ignored, and "
      "oldest rows are evicted above 30 facts per user; individual facts are limited to 200 "
      "characters. Direct recall reads the store without generation. Project-related search "
      "can incorporate accepted profile facts when the question refers to the user's work.")
    table([["Refused content", "Reason for refusal"],
           ["Secrets, account/contact identifiers, emails, long digit runs", "No tutoring benefit; high exposure cost"],
           ["Health/protected attributes, finances, grades, misconduct", "Sensitive information not needed for teaching"],
           ["Third-party/quoted claims, hypotheticals, negation, temporary states", "Not reliable, durable facts about this user"],
           ["Opt-outs and recognizable instruction injection", "Respect retention choices; limit stored-prompt injection"]],
          [280, 234])
    p("<b>Isolation and control.</b> An opt-out applies to the entire message and overrides "
      "an explicit request to remember. A sensitive sentence is vetoed even if it also contains "
      "a safe preference. History is scoped by (user_id, session_id), limited to three "
      "exchanges, and refused questions/responses are redacted before replay. Category "
      "forgetting and full reset also clear that user's histories, preventing deleted facts "
      "from returning through conversational context. A fresh session retains only that user's "
      "accepted profile. The optional memory_path persists facts across restarts; otherwise "
      "the store is in process memory.")
    p("<b>Limits.</b> English regexes can miss sensitive paraphrases or reject harmless research "
      "topics; passing negative probes is not a privacy proof. Preferences can become "
      "contradictory because they are not replacement categories. SQLite is unencrypted and "
      "user_id is trusted rather than authenticated. Refusal prevents retained profile facts "
      "and replayed turns; the current message can still reach the configured endpoint.")
    story.append(PageBreak())

    heading("5. Validation, provenance and reproducibility")
    table([["Evidence", "Result and status"],
           ["Archived real-model ablation", "15 follow-ups + 5 controls; 0/40 failed condition runs"],
           ["New local 8B dev run", "10/10 single-turn and 3/3 dialogue endings at full recall; all 12 memory probes pass (4 positive, 8 negative)."],
           ["Offline validation", "49/49 tests; mock dev: 100% single-turn and dialogue-ending recall; all public and extra memory probes checked."],
           ["Local evidence-aware self-judge", "20 actual answers judged; 18 valid JSON replies, 2 malformed replies retained. One LoRA error documented; three identical requests all get 5/5."],
           ["Official course judge / quality bar", "Not run. Local self-judge scores do not establish the 0.76 quality threshold or official bonus eligibility."]],
          [195, 319])
    p("The {}-test suite passes offline in Python {} and covers contracts, citations, "
      "read-only corpus access, thread handoff, endpoint failures, isolation, persistence "
      "and privacy controls. Thread handoff does not establish safe concurrent writes. "
      "Earlier real dev repeats reportedly had equal scores but a citation-order change; "
      "their raw repeat logs are unavailable. The current local study retains exact "
      "answers, passages, judge requests/responses and model/server configuration.".format(
          data["offline_tests"], data["runtime"]["python"]))
    p("<b>Reproduce.</b> Install requirements.txt and supply the endpoint/model explicitly. "
      "local_inference/README.txt gives the cached checkpoint revision and local server "
      "command. The new judge protocol is fixed before generating the 20 public answers. "
      "Checksums and raw calls are saved under report/bonus/. The older rewriting "
      "ablation lacks run-time code/corpus hashes and raw answers; those were not "
      "retroactively asserted as execution provenance. Use --out to preserve that archive.")
    story.append(Preformatted(
        "python -m unittest discover -s tests -v\n"
        "python eval_dev.py --mock\n"
        "python eval_dev.py --endpoint \"$A3_ENDPOINT\" --model \"$A3_MODEL\"\n"
        "python ablation/run_ablation.py --endpoint \"$A3_ENDPOINT\" \\\n"
        "  --model \"$A3_MODEL\" --split heldout --out /tmp/real-ablation.json\n"
        "python local_inference/run_judge_study.py --endpoint \"$A3_ENDPOINT\" \\\n"
        "  --model \"$A3_MODEL\" --output /tmp/new-local-judge-study\n"
        "python report/build_report.py", code))
    heading("6. Observed local judge error (bonus evidence)")
    p("For dev-st-06, the actual LoRA answer says <i>'This is in contrast to full fine-tuning, "
      "which requires'</i> and gives <font face='Courier'>L_Adpt(2dr+r+d)+2L_LN*d</font> "
      "(ASCII notation), the adapter/LayerNorm parameter-count formula. "
      "The supplied LoRA passage explicitly defines its L_Adpt term as the number of "
      "<b>adapter layers</b>; the supplied abstract says full fine-tuning <b>retrains all "
      "model parameters</b>. Thus the answer wrongly attributes an adapter formula to "
      "full fine-tuning. Yet the evidence-aware local 8B judge scores accuracy, relevance, "
      "completeness and grounding <b>5/5</b>, reports no unsupported claims, and repeats "
      "those scores on <b>all three identical requests</b>. The contradiction was in its "
      "input, not withheld evidence. Full answer, exact formula, source chunks, fixed "
      "judge prompt and unedited responses are in report/bonus/local_lora_case.json. "
      "This case was selected from the complete saved 20-answer run; no bot answer was "
      "edited. It demonstrates a local self-judge failure, not an official grader error "
      "or population error rate. No hidden prompt was reconstructed; staff determine "
      "bonus eligibility. Retrieval recall alone cannot detect this factual error.")
    heading("7. AI-use disclosure")
    p("The prior team draft documents Claude Code (Anthropic) assistance with core "
      "retrieval/memory code, prompts, tests, PACE experiments, analysis and the initial "
      "report. OpenAI Codex assisted with compatibility/privacy fixes, local validation "
      "and tooling in the earlier revision. For this revision, Codex pulled and inspected "
      "the collaborator's changes, checked archived per-dialogue outcomes against their "
      "labels, recomputed statistics, replayed the mock diagnostic, ran the offline suite, "
      "set up local 8B inference, captured the judge study and checked the LoRA claim "
      "against the actual supplied passages. The archived ablation came from the "
      "collaborator; the new local judge run was executed for this revision. Codex also "
      "wrote/rendered this report. Verification is automated and linked to "
      "saved artifacts; no undocumented independent human review or course-judge evaluation "
      "is represented as completed.")

    def footer(canvas, doc):
        canvas.setFillColor(colors.HexColor("#536b7a"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(0.6 * inch, 0.28 * inch, "ECE 6256 / CS 6604 · A3 · Revised report")
        canvas.drawRightString(7.9 * inch, 0.28 * inch, str(doc.page))

    document = SimpleDocTemplate(str(HERE / "report.pdf"), pagesize=letter,
                                 leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                                 topMargin=0.48 * inch, bottomMargin=0.5 * inch,
                                 title="The Course TA-Bot: Retrieval and Memory",
                                 author=authors, pageCompression=1)
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", default="", help="Optional full names supplied by the team")
    args = parser.parse_args()
    data = analyze()
    make_plot(data)
    build(data, args.authors)
    print("Rebuilt report/report.pdf and its checked evidence/figures.")


if __name__ == "__main__":
    main()
