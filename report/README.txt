Revised A3 report, November 6, 2026

report.pdf is the three-page report. build_report.py is its editable source.
ablation_v2.png and ablation_v2.pdf are the raster/vector versions of its figure.
analysis.json records recomputed counts, paired statistics, Wilson intervals,
mechanism checks, environment information, source revision, and SHA-256 hashes.

Evidence used

* ../ablation/results.json: collaborator's archived Llama-3.1-8B-Instruct run.
  The original results and original ablation plot are preserved unchanged.
* ../ablation/dialogues.jsonl: authored tuning and labelled held-out dialogues.
* validation/mock_heldout.json: fresh held-out replay with the bundled mock.
  This diagnoses subject carry-over; it is not a real-model answer evaluation.
* validation/unit_tests.txt: all 49 current offline tests pass.
* validation/dev_eval_mock.txt: fresh public mock self-check.

The previous report at Git commit 3a7353c describes A40/vLLM hardware, real-model
public-dev scores, repeat runs and a stand-in answer scorer. Their raw logs and
answers are not committed. The revision labels these prior reported checks and
does not turn them into newly verified results or a course-judge quality score.
The old PDF remains available in Git history.

Reproduce the local evidence (Python 3.10+ for the current test suite)

  python -m pip install -r requirements.txt -r report/requirements.txt
  python -m unittest discover -s tests -v > report/validation/unit_tests.txt 2>&1
  python eval_dev.py --mock > report/validation/dev_eval_mock.txt
  python ablation/run_ablation.py --mock --split heldout --out report/validation/mock_heldout.json
  python report/build_report.py

Add an author line when the full team names are available:

  python report/build_report.py --authors "Full name and Full name"

The builder checks per-dialogue hits against citations and gold labels, verifies
the saved summary and exact McNemar p-values, and compares the real and mock
recovery positions. It uses Wilson intervals in the revised figure because the
original percentile bootstrap is degenerate for the five perfect controls.
The paired gain interval uses 2,000 bootstrap samples, seed 42.

A real-model rerun needs an available endpoint and model name. Save any new run
to another path with --out; do not overwrite ../ablation/results.json unless
intentionally replacing the archived experiment. No new real-model run was
performed for this report revision.
