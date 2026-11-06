"""Grading-style checks, kept out of the submission.

Offline by default. To run the same checks on a real model:
    A3_ENDPOINT=http://localhost:11434/v1 A3_MODEL=llama3.1-8k python -m unittest tests.test_grading
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mock_endpoint import start_mock_endpoint, stop_mock_endpoint  # noqa: E402
from ta_bot import TABot  # noqa: E402

REAL_ENDPOINT = os.environ.get("A3_ENDPOINT")
REAL_MODEL = os.environ.get("A3_MODEL", "")
PROBES = ROOT / "tests" / "memory_probes.jsonl"
DEV = ROOT / "dev_eval"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = None
        if REAL_ENDPOINT:
            cls.endpoint, cls.model = REAL_ENDPOINT, REAL_MODEL
        else:
            cls.endpoint, cls.server = start_mock_endpoint()
            cls.model = "mock-model"

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            stop_mock_endpoint(cls.server)

    def config(self, corpus=None):
        # Exactly the five keys the README says the harness sets.
        return {"endpoint": self.endpoint, "model": self.model,
                "corpus_path": str(corpus or ROOT / "corpus"),
                "temperature": 0.0, "seed": 42}

    def bot(self, corpus=None):
        return TABot(self.config(corpus))


class Contract(Base):
    def test_signatures_match_the_readme(self):
        wanted = {"__init__": ["self", "config"],
                  "chat": ["self", "session_id", "user_id", "message"],
                  "reset_memory": ["self", "user_id"],
                  "get_memory": ["self", "user_id"]}
        for name, params in wanted.items():
            self.assertEqual(list(inspect.signature(getattr(TABot, name)).parameters), params)

    def test_paper_answer_shape_and_real_paper_ids(self):
        conn = sqlite3.connect(str(ROOT / "corpus" / "corpus.db"))
        ids = {row[0] for row in conn.execute("SELECT paper_id FROM papers")}
        conn.close()
        result = self.bot().chat("s", "u", "What is BERT and what are its objectives?")
        self.assertEqual(set(result), {"answer", "citations"})
        self.assertTrue(result["answer"].strip())
        self.assertTrue(result["citations"])
        self.assertEqual(len(result["citations"]), len(set(result["citations"])))
        self.assertLessEqual(set(result["citations"]), ids)

    def test_get_memory_returns_strings_and_unknown_user_is_safe(self):
        bot = self.bot()
        bot.reset_memory("nobody")
        self.assertEqual(bot.get_memory("nobody"), [])

    def test_reset_empties_store(self):
        bot = self.bot()
        bot.chat("s", "u", "My name is Jordan.")
        self.assertTrue(bot.get_memory("u"))
        bot.reset_memory("u")
        self.assertEqual(bot.get_memory("u"), [])

    def test_runs_from_another_directory(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            os.chdir(folder)
            try:
                self.assertTrue(self.bot().chat("s", "u", "What is RAG?")["citations"])
            finally:
                os.chdir(previous)

    def test_chat_works_from_a_different_thread(self):
        bot = self.bot()
        outcome = {}

        def run():
            try:
                outcome["result"] = bot.chat("s", "u", "My name is Jordan. What is RAG?")
                outcome["memory"] = bot.get_memory("u")
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = repr(exc)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        self.assertNotIn("error", outcome, outcome.get("error"))

    def test_read_only_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder) / "corpus"
            corpus.mkdir()
            shutil.copy(str(ROOT / "corpus" / "corpus.db"), str(corpus / "corpus.db"))
            (corpus / "corpus.db").chmod(stat.S_IRUSR)
            corpus.chmod(stat.S_IRUSR | stat.S_IXUSR)
            try:
                self.assertTrue(self.bot(corpus).chat("s", "u", "What is RAG?")["citations"])
            finally:
                corpus.chmod(stat.S_IRWXU)

    def test_unusual_messages_never_crash(self):
        bot = self.bot()
        for message in ("", "   ", "\n\t", "?", "!!!", "'", '"', "*", "AND OR NOT", 'NEAR("a" "b")',
                        "title:bert", "'; DROP TABLE papers; --", "🤖" * 20, "résumé naïve 日本語",
                        "word " * 4000, "a" * 20000, "\x00 null"):
            with self.subTest(message=message[:30]):
                try:
                    result = bot.chat("weird", "u", message)
                except TimeoutError:
                    continue  # the only failure the contract allows
                self.assertIsInstance(result["answer"], str)
                self.assertIsInstance(result["citations"], list)


class Submission(unittest.TestCase):
    source = (ROOT / "ta_bot.py").read_text()

    def test_nothing_hard_coded(self):
        self.assertIsNone(re.search(r"https?://|localhost|\d+\.\d+\.\d+\.\d+", self.source))
        self.assertIsNone(re.search(r"(?i)llama-?\d|gpt-oss|mock-model", self.source))
        self.assertIsNone(re.search(r"/(?:Users|home|storage)/", self.source))

    def test_no_paper_ids_in_code(self):
        self.assertEqual(re.findall(r"\bL\d{2}-\d{2}\b", self.source), [])

    def test_imports_are_declared(self):
        imports = set(re.findall(r"^(?:from|import) (\w+)", self.source, re.M))
        declared = (ROOT / "requirements.txt").read_text().lower()
        for name in imports - set(sys.stdlib_module_names) - {"__future__"}:
            self.assertIn(name.lower(), declared)

    def test_no_external_api_clients(self):
        self.assertIsNone(re.search(r"(?i)\bimport\s+(openai|anthropic)|api\.openai|api_key", self.source))


class Memory(Base):
    def check(self, probe):
        bot = self.bot()
        plant, test = probe["plant_session"], probe["test_session"]
        expected = probe.get("expected_facts") or [probe["expected_fact"]]
        for user in {plant["user_id"], test["user_id"]}:
            bot.reset_memory(user)
        for turn in plant["turns"]:
            bot.chat(plant["session_id"], plant["user_id"], turn["content"])
        store = " ".join(bot.get_memory(test["user_id"])).lower()
        answer = ""
        for turn in test["turns"]:
            answer = bot.chat(test["session_id"], test["user_id"], turn["content"])["answer"].lower()
        return store, answer, [fact.lower() for fact in expected]

    def test_all_probes(self):
        for probe in jsonl(PROBES) + jsonl(DEV / "memory_probes.jsonl"):
            with self.subTest(probe=probe["id"]):
                store, answer, facts = self.check(probe)
                for fact in facts:
                    if probe["type"] == "positive":
                        self.assertIn(fact, store)
                        self.assertIn(fact, answer)
                    else:
                        self.assertNotIn(fact, store)
                        self.assertNotIn(fact, answer)


class Reproducibility(Base):
    def run_all(self):
        bot = self.bot()
        cited = []
        for item in jsonl(DEV / "single_turn.jsonl"):
            cited.append(bot.chat("s-" + item["id"], "u", item["question"])["citations"])
        for item in jsonl(DEV / "multi_turn.jsonl"):
            for turn in item["turns"]:
                cited.append(bot.chat("m-" + item["id"], "v", turn["content"])["citations"])
        return cited

    def test_two_runs_cite_the_same_papers(self):
        self.assertEqual(self.run_all(), self.run_all())


if __name__ == "__main__":
    unittest.main()
