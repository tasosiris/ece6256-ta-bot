# Assignment 3 (A3): The Course TA-Bot — Conversational RAG with Memory

Build a teaching-assistant chatbot that retrieves from a corpus of course papers, answers
questions with grounded citations, and remembers facts about each user across sessions.

| | | | |
|---|---|---|---|
| **Due** | Mon, November 2, 11:59 pm ET |  |  |
| **Work mode** | Paired (teams of 2) | **Effort** | ≈25–30 hours |
| **Code** | Canvas (zip) | **Report** | Gradescope (PDF, ≤3 pages) |

> In Assignment 2 you taught a model *when* to ask versus answer. Here you build a full
> conversational system that answers from evidence, cites its sources, and learns what each
> user cares about — without memorizing things it should forget. The pipeline (chunking,
> indexing, retrieval, generation, memory) is yours to design; the interface contract and
> evaluation harness are fixed.

---

## Setup

```bash
pip install -r requirements.txt
```

> **On PACE-ICE:** your home directory is small. Point your Hugging Face cache at the shared
> read-only copy and run offline — no download needed:
>
> ```bash
> export HF_HOME=/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache
> export HF_HUB_OFFLINE=1
> export TRANSFORMERS_OFFLINE=1
> ```
>
> Available models: `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.1-8B-Instruct`,
> `openai/gpt-oss-20b`.
> The generation and embedding endpoint is the **course vLLM server** (see "Endpoint" below) —
> you do **not** load the model yourself for inference.

---

## Endpoint

All generation and embedding calls go through a single **vLLM-compatible OpenAI API** endpoint.

- **For graded runs** on PACE-ICE the TAs start a shared vLLM server. Your code receives the URL
  via `config["endpoint"]` — never hard-code it.
- **For local development** use the bundled mock endpoint:

```bash
python mock_endpoint.py                    # starts on a random port, prints URL
# or use eval_dev.py --mock (starts it for you)
```

The mock returns canned completions and deterministic embeddings (SHA-256-seeded, 128-dim).
It is good enough to verify your pipeline wiring — retrieval, memory, output format — before
you test on the real model.

---

## Starter Kit Contents

| File | Description |
|---|---|
| `ta_bot.py` | **Your main file.** Implements `TABot` — the class the harness imports. |
| `mock_endpoint.py` | Local mock vLLM server for offline testing. |
| `eval_dev.py` | Dev-set self-check script (see "Self-Check" below). |
| `corpus/` | Course paper corpus — `corpus.db` (SQLite with papers, chunks, FTS5 index). |
| `dev_eval/` | Small public dev set (single-turn, multi-turn, memory probes). |
| `requirements.txt` | Pinned dependencies. |
| `README.md` | This file. |

---

## What to Build

You implement the four methods of `TABot` in `ta_bot.py`. Do **not** rename the class or
change the method signatures — the autograder imports your file by name and calls these
methods directly.

### The interface contract

```python
class TABot:
    def __init__(self, config: dict) -> None:
        """
        Required config keys (set by the harness — do not hard-code):
            endpoint    (str)   — vLLM-compatible base URL
            model       (str)   — model name served at the endpoint
            corpus_path (str)   — path to the paper corpus directory
            temperature (float) — 0 for graded runs
            seed        (int)   — fixed seed for reproducibility
        You may add your own config keys for custom settings.
        """

    def chat(self, session_id: str, user_id: str, message: str) -> dict:
        """
        Returns {"answer": str, "citations": [paper_id, ...]}
        - answer:    natural-language response
        - citations: paper_id strings from the corpus papers table (paper-level, never chunk ids)
        """

    def reset_memory(self, user_id: str) -> None:
        """Empty the persistent memory store for this user."""

    def get_memory(self, user_id: str) -> list[str]:
        """Return stored facts about this user as readable strings (read-only view)."""
```

### Session vs. user semantics

- `session_id` scopes **conversational state** (history, anaphora). A new `session_id` starts a
  fresh conversation.
- `user_id` scopes **cross-session memory**. The same `user_id` across different sessions shares
  memory; a new `user_id` is a stranger.

### Error contract

- If the endpoint is unreachable or slow, raise `TimeoutError`. The harness catches it and marks
  that turn as failed — it does not kill your process.
- Any other unhandled exception scores zero for that turn.

---

## Your Design Space

The following are **your decisions** — there is no single correct answer:

1. **Chunking strategy** — how you split papers into pieces for indexing. Chunk size, overlap,
   and boundaries are free variables. Scoring is at paper level (INV-8), so your chunk design
   won't be penalised directly — only retrieval effectiveness matters.

2. **Retrieval method** — BM25, dense embeddings via the endpoint, hybrid, re-ranking, or
   anything else. The endpoint serves `/v1/embeddings` alongside `/v1/chat/completions`.

3. **Memory architecture** — how you store, update, and filter cross-session facts. The
   memory probes test both what you store (positive recall) and what you refuse to store
   (negative probes). Design accordingly.

4. **Prompt engineering** — system prompt, few-shot examples, chain-of-thought — all up to you,
   as long as they go through `config["endpoint"]`.

5. **Additional dependencies** — add to `requirements.txt`. Stick to standard PyPI packages.

---

## Self-Check (Dev Set)

A small public dev set ships in `dev_eval/`. Run the self-check locally:

```bash
# With the mock endpoint (offline, no GPU):
python eval_dev.py --mock

# With a real vLLM endpoint:
python eval_dev.py --endpoint http://localhost:8000/v1
```

The script reports:
- **Single-turn retrieval hit-rate** — did your citations include the gold paper?
- **Multi-turn retrieval hit-rate** — same, across multi-turn dialogues.
- **Memory probe pass/fail** — did your memory store the expected facts?

> **The dev set is small on purpose.** High dev scores do not guarantee high graded scores.
> The hidden test set covers different papers, conversation patterns, and memory scenarios
> than the dev set. Build a system that generalises, not one that overfits to these examples.

---

## Grading

### Pre-submission sanity check

Before submitting, run `eval_dev.py --mock` — it includes preflight checks that verify
your interface contract (import, `chat()` shape, `reset_memory()`, `get_memory()` return
type) before running the dev eval. Fix any preflight failures before submitting.

### Graded evaluation (PACE-ICE)

The TAs run your submission against the hidden test set with the real vLLM endpoint and
an LLM judge. Scores are uploaded to Canvas manually.

**Headline metrics:**

| Metric | What it measures |
|---|---|
| **Retrieval hit-rate** | % of questions where your citations include the gold paper |
| **Answer quality** | LLM-judge 1–5 score (accuracy, relevance, completeness, grounding) |
| **Memory recall** | % of positive probes where the planted fact is recalled |
| **Negative probe pass** | Must-not-store facts do not leak into memory or output |

**Thresholds** _(calibrated 2026-10-23)_:

| Metric | Passing Threshold | Notes |
|---|---|---|
| retrieval_hit_rate | **0.70** | Basic retrieval bar (FTS alone easily clears) |
| answer_quality | **0.76** | Primary discriminator — grounded, accurate answers |
| memory_recall | **0.80** | Allows 1 probe miss; negative_probe is the isolation gate |
| negative_probe | **pass** | Binary — must-not-store facts must not leak |

A naive baseline that does minimal retrieval and no memory filtering sets the floor; a
well-built system sets the ceiling. You need to meaningfully beat naive — not be perfect.

**Reproducibility:** with `temperature=0` and the fixed seed, your scores must reproduce
within tolerance across two runs. Large variance suggests non-determinism in your pipeline
(random chunking, unseeded sampling, etc.) — fix it.

**Judge caveat (on purpose):** you are being graded by an LLM judge two weeks before
Lecture 12 teaches you why that is dangerous. Up to **5 bonus points** for documenting a
case where the judge scored your system wrongly — with evidence.

---

## Grading Rubric

| Component | What full credit looks like | Points |
|---|---|---|
| **Harness scores** | Hit-rate and judged answer quality above posted thresholds; memory probes pass. | 35 |
| **Ablation + analysis** | One axis, honest measurement, a plot, and an interpretation that would survive a skeptical reviewer. | 35 |
| **Memory design writeup** | What you store, when you write, what you refuse to store and why. Privacy reasoning counts. | 15 |
| **Reproducibility** | Clean clone + endpoint reproduces harness scores within tolerance. | 10 |
| **AI-Use Disclosure** | Complete and candid. | 5 |
| **Total** | | **100** |

---

## Submission Checklist

Before submitting, verify:

- [x] `python eval_dev.py --mock` runs without errors
- [x] `chat()` returns the correct schema (`{"answer": str, "citations": [paper_id, ...]}`)
- [x] Citations are `paper_id` strings from `corpus.db` (papers table), not chunk ids or filenames
- [x] `reset_memory()` + `get_memory()` work correctly
- [x] `requirements.txt` lists all runtime dependencies (report tools are in `requirements-dev.txt`)
- [x] No hard-coded endpoints, file paths, or model names in `TABot` — everything comes from `config`

### What to submit

**1. Code (Canvas)** — Upload your project as a zip. Must include `ta_bot.py` at the root,
plus `requirements.txt` with all dependencies. The TAs run your code on PACE-ICE with the
hidden test set and real vLLM endpoint.

**2. Report (Gradescope, PDF, ≤3 pages)** — Must include:

- **Ablation study** (R4): one axis (rewriting on/off, k, chunk size, embedding model, etc.),
  honest measurement, a plot, and an interpretation that would survive a skeptical reviewer.
- **Memory design writeup** (R3): what you store, when you write, what you deliberately refuse
  to store and why. Privacy reasoning counts.
- **AI-Use Disclosure** (≤half page, included in the 3-page limit): which AI tools you used,
  for what, and how you verified the results. A candid disclosure never costs you points on
  the work itself; an empty or evasive disclosure on work that plainly used AI is an
  honor-code matter.

---

## Academic Integrity

- Build your own pipeline. You may use any libraries, but the retrieval, memory, and
  prompting logic must be your own work.
- Do **not** reverse-engineer the hidden test set or the judge prompt.
- Do **not** hard-code answers or paper ids — the grader detects this.
- Do **not** use external APIs (OpenAI, Anthropic, etc.) — all calls go through the
  course vLLM endpoint only.

---

## Implementation Notes

`ta_bot.py` implements `TABot`: BM25 retrieval over `corpus.db`, query rewriting for follow-ups,
per-session history, and a filtered per-user memory store. Python 3.8 or newer is supported.
The supplied chunking is used without rebuilding the corpus. Generation requests contain the
configured model, temperature, and seed. Network errors and malformed or empty completions
raise `TimeoutError`.

Optional config keys:

| Key | Default | Purpose |
|---|---|---|
| `rewrite` | `True` | Rewrite anaphoric follow-ups and retain their last subject. Explicit comparisons also retain earlier primary papers. |
| `timeout` | `60` | Timeout in seconds for each endpoint request. |
| `memory_path` | `":memory:"` | SQLite file for accepted user facts. Use a file to retain memory across process restarts. |
| `top_papers` | `3` | Papers selected per retrieval query; contextual carry-over may add papers. This is the ablation axis. |

Memory accepts explicit first-person names, projects, interests, academic backgrounds, and
response preferences. It refuses opt-outs, sensitive information, third-party and quoted
statements, hypotheticals, questions, negations, and temporary states. New names/projects
replace previous values; other facts are deduplicated, with a maximum of 30 per user.
History is limited to three exchanges per `(user_id, session_id)`. Refused statements and
their responses are redacted before future replay. `reset_memory()` and explicit forgetting
also clear the user's histories; category deletion preserves other stored facts.
`get_memory()` returns a read-only copy. Call the optional `close()` when finished with a bot.

Filtering uses conservative English patterns and can miss paraphrases or reject harmless
academic language. SQLite files are ordinary, unencrypted files. The current user message
still goes to the configured course endpoint when generation is needed. Returned citations
identify evidence supplied to generation; they do not independently verify generated claims.

### Reproduce the completed local run

```bash
python -m pip install -r requirements-dev.txt
python scripts/verify.py
python experiments/ablation.py --mock
python scripts/build_report.py
python scripts/build_submission.py
```

`requirements-lock.txt` records the exact packages tested in a fresh Python 3.8 environment.
For that environment, install it instead of the version ranges in `requirements-dev.txt`.
Only `requirements.txt` is needed to run the bot or the standard self-check.

The checks report 35 passing regression tests, 100% retrieval recall on 10 single-turn
questions and 3 dialogue endings, and 12/12 memory probes (2 public positives, 2 additional
positives, and 8 authored negatives). The evaluation checks **all** `expected_facts` when
provided, and checks the test user's store for cross-user isolation probes.

The ablation varies `top_papers` over 1, 2, 3, 5, and 8, with two fresh runs at each setting.
All settings achieve 100% public retrieval recall, including all 10 dialogue turns. Mean
evidence size rises from 4,242 to 14,178 characters. This small set shows no recall benefit
from larger k; it does not establish answer quality or the best k on unseen questions.
The existing default of 3 is retained. Repeated mock answers and citations match exactly.

Generated deliverables:

| Path | Contents |
|---|---|
| `results/ablation.json` | Raw questions, gold labels, answers, citations, timings, evidence sizes, repeats, and source/data hashes. Gold labels are used only after `chat()` for scoring. |
| `results/ablation.png`, `results/ablation.pdf` | Exportable ablation plot. |
| `results/verification.json`, `results/unit_tests.txt`, `results/dev_eval.txt` | Test metadata and full logs. |
| `report/report.pdf` | Three-page report with the study, memory design, limitations, and AI-use disclosure. |
| `dist/ece6256-ta-bot.zip` | Code submission with `ta_bot.py` at the root, corpus, tests, scripts, and report; excludes virtual environments, Git history, and runtime memory files. |

The report builder checks the measured implementation's hash before generating the PDF.
The zip is created from explicit project paths. Extract it into an empty directory and run
`python eval_dev.py --mock` to check the submission independently of the checkout.

### Rewriting ablation kit

`experiments/ablation.py` varies `top_papers`, which the public dev set cannot separate.
`ablation/` compares context handling on and off (`rewrite`) on questions built to separate
them: 30 follow-ups and 10 standalone controls, split into `tune` and `heldout`. Adjust the bot
on `tune` and report `heldout`. It needs a real model, because the mock cannot rewrite.

```bash
python ablation/run_ablation.py --endpoint URL --model NAME --split heldout
python ablation/plot.py                     # writes ablation/ablation.png
```

To develop against a local model, use the context-size override in `ablation/Modelfile`, because
Ollama silently cuts long prompts at its default size:

```bash
ollama pull llama3.1:8b
ollama create llama3.1-8k -f ablation/Modelfile
python eval_dev.py --endpoint http://localhost:11434/v1 --model llama3.1-8k
```

### Configure the course endpoint

The server is provided by the course staff; this project does not launch it. Once you have
its URL, save the connection with:

```bash
python scripts/configure_endpoint.py --endpoint "$COURSE_ENDPOINT"
```

The setup script checks `/v1/models`, detects the model if exactly one is served, and saves
the validated settings in `course_endpoint.local.json` (gitignored and excluded from the
submission archive). If several models are served, add `--model "$COURSE_MODEL"` to select
one. It accepts URLs with or without a `/v1` suffix. It only saves after a successful check.
No course connection is created until you supply a real server URL.

After setup, run:

```bash
python eval_dev.py
python experiments/ablation.py
```

The ablation saves real-endpoint results to `results-course/` by default. Both tools resolve
settings from command-line arguments first, then `COURSE_ENDPOINT`/`COURSE_MODEL`, then the
local configuration file. If the model is omitted, they query `/v1/models` to detect it.
`--mock` continues to select the bundled mock explicitly. The harness still passes its own
configuration directly to `TABot`, independently of this local CLI setup.

You can also supply the URL and served model explicitly without saving a local file:

```bash
python eval_dev.py --endpoint "$COURSE_ENDPOINT" --model "$COURSE_MODEL"
python experiments/ablation.py --endpoint "$COURSE_ENDPOINT" --model "$COURSE_MODEL" --output results-course
```

These variables must be set to the URL and model provided by the course staff. Real-model
results should remain separate from the mock artifacts. No course endpoint was available
for the recorded run: judged answer quality, hidden-set scores, and real-model leakage
remain unmeasured. The mock returns canned text, so its retrieval and memory checks cannot
establish the real-model answer-quality threshold.
