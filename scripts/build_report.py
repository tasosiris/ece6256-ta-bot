#!/usr/bin/env python3
"""Build the three-page report from measured results, without a LaTeX install."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                               SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = json.loads((ROOT / "results/ablation.json").read_text())
    checks = json.loads((ROOT / "results/verification.json").read_text())
    for record in (results["metadata"], checks):
        expected = record["sha256"]["ta_bot.py"]
        actual = hashlib.sha256((ROOT / "ta_bot.py").read_bytes()).hexdigest()
        if expected != actual:
            raise SystemExit("ta_bot.py changed: rerun verification and ablation before building the report")
    if not checks["unit_tests"]["passed"] or not checks["dev_eval"]["passed"]:
        raise SystemExit("Verification must pass before building a report")
    if results["metadata"]["mode"] != "mock":
        raise SystemExit("This report's discussion describes the bundled mock; revise it for a real-model run")
    settings = results["settings"]
    by_k = {setting["top_papers"]: setting for setting in settings}
    if not {1, 3, 8} <= set(by_k):
        raise SystemExit("The report compares k=1, 3, and 8; include those settings")
    first, default, last = (by_k[k]["metrics"] for k in (1, 3, 8))
    metadata = results["metadata"]
    tests = checks["unit_tests"]["run"]
    memory = checks["dev_eval"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyReport", fontName="Helvetica", fontSize=9.4,
                              leading=12.4, spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=19,
                              leading=23, textColor=colors.HexColor("#173a53"), spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportHeading", fontName="Helvetica-Bold", fontSize=11.5,
                              leading=15, spaceBefore=9, spaceAfter=6,
                              textColor=colors.HexColor("#173a53")))
    styles.add(ParagraphStyle(name="CaptionReport", fontName="Helvetica", fontSize=8,
                              leading=10, spaceAfter=7))
    styles.add(ParagraphStyle(name="CodeReport", fontName="Courier", fontSize=7.6,
                              leading=10, spaceAfter=8))
    story = []

    def p(text):
        story.append(Paragraph(text, styles["BodyReport"]))

    def heading(text):
        story.append(Paragraph(text, styles["ReportHeading"]))

    def table(data, widths):
        item = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f5")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEADING", (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#7993a5")),
            ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.HexColor("#b7c8d3")),
        ]))
        story.append(item)
        story.append(Spacer(1, 7))

    story.append(Paragraph("The Course TA-Bot", styles["ReportTitle"]))
    p("ECE 6256 / CS 6604 · Assignment 3 · Conversational retrieval with filtered user memory")
    heading("1. System design")
    p("The bot searches 108 course papers and 5,711 supplied chunks through SQLite FTS5. "
      "Quoted, stopword-filtered query terms form a lexical OR search. BM25 gives title terms "
      "weight 5; the top 40 chunks are grouped into papers, with up to two chunks per paper. "
      "Paper scores sum the selected chunk scores. Sorting uses explicit identifiers to break ties.")
    p("For follow-ups, a course-endpoint completion rewrites references into a standalone query. "
      "Reciprocal rank fusion combines original and rewritten results (damping 60), retaining "
      "the original ranking on ties. Pronouns retain the last subject; explicit comparisons can "
      "add earlier primary papers. Topic switches skip rewriting. The last three exchanges "
      "provide context. Generation receives attributed passages, instructions to cite paper IDs, "
      "and a separate description of stored user facts for personalization. All inference goes "
      "through the configured endpoint; no external model API or local inference is used.")
    heading("2. Controlled ablation: papers retrieved per query")
    p("Only retrieval depth k changes (1, 2, 3, 5, 8). The corpus, chunking, BM25 weights, "
      "prompts, rewriting, seed 42, and temperature 0 remain fixed. Each setting runs twice "
      "with fresh bot state on the 10 public single-turn questions and 3 dialogues (10 turns). "
      "Gold paper IDs are consulted after generation only. Dialogue-end recall matches "
      "eval_dev.py; all-turn recall is an additional diagnostic. Evidence size includes passage "
      "headers and measures prompt cost in characters, rather than model tokens.")
    story.append(Image(str(ROOT / "results/ablation.png"), width=7.15 * inch, height=2.10 * inch))
    story.append(Paragraph("Figure 1. Bundled mock endpoint. Curves overlap at 100% recall; larger k "
                           "increases evidence size. Prior subjects can make returned paper counts exceed k. "
                           "The mock generates canned text and provides no answer-quality measurement.",
                           styles["CaptionReport"]))
    data = [["k", "Single recall", "Dialogue final", "All dialogue turns", "Mean evidence chars"]]
    for setting in settings:
        metric = setting["metrics"]
        data.append([str(setting["top_papers"]), "{:.0%}".format(metric["single_hit_rate"]),
                     "{:.0%}".format(metric["multi_final_hit_rate"]),
                     "{:.0%}".format(metric["multi_all_turn_hit_rate"]),
                     "{:,.0f}".format(metric["mean_evidence_chars"])])
    table(data, [30, 82, 88, 115, 153])
    story.append(PageBreak())

    heading("3. Interpretation and limits of the ablation")
    p("The public retrieval task is already saturated at k=1: every tested setting achieves "
      "100% single-turn recall, dialogue-ending recall, and all-turn dialogue recall. Mean "
      "evidence grows from {:,.0f} characters at k=1 to {:,.0f} at k=8 ({:.2f} times). "
      "At the existing default k=3, the mean is {:,.0f} characters. These observations show "
      "additional prompt cost without a measured public-set recall benefit. They do not "
      "establish which k produces better explanations.".format(
          first["mean_evidence_chars"], last["mean_evidence_chars"],
          last["mean_evidence_chars"] / first["mean_evidence_chars"], default["mean_evidence_chars"]))
    p("The existing default k=3 is retained as an unvalidated recall cushion, not an empirically "
      "optimal choice. A real-endpoint comparison should judge answer accuracy and grounding "
      "at k=1 versus k=3 before choosing between them. The sample has only 10 independent "
      "single-turn items and 3 independent dialogues; dialogue turns are correlated. Broad "
      "lexical matching plus context carry-over can supply irrelevant papers while still "
      "satisfying recall. Returned citations list evidence supplied to generation; their "
      "presence alone does not prove that generated claims used or correctly interpreted it.")
    heading("4. Memory design and privacy reasoning")
    p("<b>What is stored.</b> A per-user SQLite table stores readable category/fact pairs. "
      "An allowlist accepts explicit first-person statements about a name, project or thesis, "
      "academic background, interests, and response preferences. Storage occurs on user turns "
      "before retrieval, never from assistant text or paper passages. Duplicate facts are ignored. "
      "New names and projects replace older values; interests, preferences, and backgrounds "
      "can coexist. At most 30 facts per user are retained, evicting older rows when needed. "
      "Without memory_path, the SQLite store lives in process memory. Setting memory_path "
      "retains facts across process restarts; session history remains process-local.")
    p("<b>What is refused.</b> A whole-message opt-out overrides extraction. Per-sentence vetoes "
      "reject secrets and identifiers, contact details, health and protected attributes, financial "
      "and academic-standing information, third-party statements, hypothetical or negated "
      "claims, temporary states, questions, and recognizable instruction injection. Anchoring "
      "patterns to first-person clauses prevents quoted third-party examples from becoming "
      "user identity. Filtering a whole sensitive sentence favors privacy over remembering a "
      "safe preference mixed into that sentence. Explicit requests to remember something "
      "do not override these refusals.")
    p("<b>Isolation and deletion.</b> Facts are keyed by user_id, while history uses both user_id "
      "and session_id. Fresh sessions share only that user's accepted facts. Direct memory "
      "questions read the store without generation, preventing invented recall. Explicit "
      "forgetting deletes the requested category or all facts and clears that user's histories, "
      "so deleted information is not replayed in later prompts. reset_memory performs the same "
      "full deletion. Refused messages and their responses are redacted from replayed history.")
    p("<b>Tradeoffs.</b> English regular expressions are conservative, not a semantic privacy "
      "guarantee. They can miss paraphrases and reject harmless academic language. Preferences "
      "can conflict because only names and projects have replacement semantics. A persistence "
      "file is ordinary SQLite, without encryption or authentication; callers must supply a "
      "trusted user identity and protect the file. Sensitive text is still sent to the configured "
      "course endpoint for its current response; refusal means no retained profile fact or "
      "replayed conversation text, not that the endpoint never sees the message.")
    story.append(PageBreak())

    heading("5. Validation and reproducibility")
    table([["Check", "Observed result"],
           ["Public single-turn retrieval", "10 / 10 gold papers found"],
           ["Public dialogue endings", "3 / 3 dialogues at full recall"],
           ["Public positive memory probes", "2 / 2; all listed expected facts checked"],
           ["Additional authored probes", "2 / 2 positive; 8 / 8 negative"],
           ["Regression suite", "{} / {} tests pass".format(tests, tests)],
           ["Ablation repeated outputs", "Identical answers, citations, evidence at every k"]],
          [225, 243])
    p("Regressions cover valid paper-level citations, deterministic retrieval, hostile FTS "
      "syntax, endpoint URL normalization, malformed replies, timeout/error conversion, "
      "follow-up subjects, topic switches, user/session isolation, memory replacement, "
      "deduplication, persistence, opt-outs, sensitive-text replay, and full/category deletion. "
      "The bundled mock's hash-to-float conversion was repaired because arbitrary float bit "
      "patterns produced NaN embeddings; it now maps integer values to bounded floats.")
    p("The measured environment uses Python {}, SQLite {}, and requests {}. The fresh virtual "
      "environment installs requirements-dev.txt; requirements-lock.txt records its exact "
      "package versions. TABot's runtime dependencies are listed separately in requirements.txt. "
      "The saved results include per-turn predictions, answers, evidence lengths, timings, "
      "environment versions, source/corpus/dataset SHA-256 checksums, and both repeated runs. "
      "No endpoint address or credential is embedded in the implementation.".format(
          escape(metadata["python"]), escape(metadata["sqlite"]), escape(metadata["requests"])))
    story.append(Preformatted(
        "python -m pip install -r requirements-dev.txt\n"
        "python scripts/verify.py\n"
        "python experiments/ablation.py --mock\n"
        "python scripts/build_report.py\n"
        "python scripts/build_submission.py", styles["CodeReport"]))
    p("To test the real course model, run eval_dev.py with --endpoint and --model, or run "
      "experiments/ablation.py with the same arguments and --output results-course. Keep "
      "real-model results separate from this mock study. Endpoint failures become TimeoutError "
      "rather than unhandled transport or response-parsing exceptions.")
    heading("6. Remaining evaluation limits")
    p("No real course vLLM endpoint was supplied during this run. Answer quality, hidden-set "
      "retrieval, real-model memory leakage, graded-server latency, and the stated 0.76 quality "
      "threshold are unmeasured. The added negative probes are authored development checks, "
      "not hidden tests. Repeated mock outputs establish local determinism, not determinism "
      "of a remote model server. No LLM judge was run, so no judge-error bonus case is claimed.")
    heading("7. AI-use disclosure")
    p("OpenAI Codex was used to inspect and extend the supplied implementation, diagnose "
      "compatibility and privacy problems, write code and regression probes, run the local "
      "evaluation and retrieval-depth ablation, generate the plot and this report, and prepare "
      "the submission archive. The supplied repository already contained retrieval, generation, "
      "rewriting, and memory code; earlier tool usage was unavailable in this session. This "
      "disclosure covers the continuation. Verification was automated through a fresh Python "
      "environment, the bundled mock, public dev data, authored tests, and repeated ablation "
      "runs. Measurements came from executed code, with raw records retained. No human "
      "review, real-model answer evaluation, or hidden-set result is represented as completed.")

    destination = ROOT / "report"
    destination.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(destination / "report.pdf"), pagesize=letter,
                            leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                            topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                            title="The Course TA-Bot: Retrieval, Memory, and Ablation",
                            author="", pageCompression=1)

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526878"))
        canvas.drawString(0.65 * inch, 0.29 * inch, "ECE 6256 / CS 6604 · A3 · Local mock evaluation")
        canvas.drawRightString(7.85 * inch, 0.29 * inch, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Built", destination / "report.pdf")


if __name__ == "__main__":
    main()
