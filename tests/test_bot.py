"""Offline tests for TABot. Run with: python -m unittest discover tests"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mock_endpoint import start_mock_endpoint, stop_mock_endpoint  # noqa: E402
from ta_bot import TABot  # noqa: E402

STORED = [
    ("My name is Alex Chen.", "name", "Alex Chen"),
    ("my name is sam", "name", "sam"),
    ("Hi, I'm Priya and I study agents.", "name", "Priya"),
    ("Call me Dana.", "name", "Dana"),
    ("I'm writing my semester project on query rewriting.", "project", "query rewriting"),
    ("My thesis is about dense retrieval, but I also like agents.", "project", "dense retrieval"),
    ("I'm working on hybrid search for my final project.", "project", "hybrid search"),
    ("I'm building a tutoring chatbot.", "project", "a tutoring chatbot"),
    ("My research focuses on speech models.", "project", "speech models"),
    ("I'm interested in memory-augmented agents.", "interest", "memory-augmented agents"),
    ("I want to learn more about reasoning models.", "interest", "reasoning models"),
    ("I care about model evaluation.", "interest", "model evaluation"),
    ("I'm a second-year PhD student in robotics.", "background", "a second-year PhD student in robotics"),
    ("I am majoring in computer engineering.", "background", "computer engineering"),
    ("I have a background in linguistics.", "background", "linguistics"),
    ("I prefer concise answers with examples.", "preference", "concise answers with examples"),
    ("Please keep answers short.", "preference", "answers short"),
]

REFUSED = [
    "My password is hunter2.",
    "My SSN is 123-45-6789.",
    "My API key is sk-abc.",
    "I was diagnosed with ADHD and I'm interested in memory.",
    "I take medication for anxiety.",
    "My GPA is 2.1.",
    "I'm failing this course.",
    "My roommate is interested in RAG.",
    "My advisor's name is Bob.",
    "Suppose my name is Bob.",
    "Imagine my project is about cats.",
    "Today I'm interested in tokenizers.",
    "My phone number is 5551234567.",
    "My email is a@b.co.",
    "I'm not interested in RAG.",
    "Is my project on RAG good?",
    "My name is Alex. Please don't remember that.",
    "Off the record, I'm interested in jailbreaks.",
    "My project is due tomorrow.",
    "What is BERT?",
]


class ScriptedHandler(BaseHTTPRequestHandler):
    """Records requests and replies with whatever the test scripted."""

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        is_rewrite = "standalone" in body["messages"][0]["content"]
        reply = self.server.rewrite_reply if is_rewrite else "answer"
        payload = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class BotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.url, self.server = start_mock_endpoint()
        self.addCleanup(self.server.server_close)
        self.addCleanup(stop_mock_endpoint, self.server)

    def bot(self, endpoint: str | None = None, **extra) -> TABot:
        return TABot(
            {
                "endpoint": endpoint or self.url,
                "model": "mock-model",
                "corpus_path": str(ROOT / "corpus"),
                "temperature": 0.0,
                "seed": 42,
                **extra,
            }
        )


class MemoryExtraction(unittest.TestCase):
    def test_stores_safe_facts(self) -> None:
        for message, category, fact in STORED:
            with self.subTest(message):
                self.assertIn((category, fact), TABot._extract_facts(message))

    def test_refuses_unsafe_facts(self) -> None:
        for message in REFUSED:
            with self.subTest(message):
                self.assertEqual(TABot._extract_facts(message), [])


class MemoryStore(BotTestCase):
    def test_new_project_replaces_old(self) -> None:
        bot = self.bot()
        bot.chat("a", "u", "My project is about RAG.")
        bot.chat("b", "u", "My project is about agents.")
        self.assertEqual(bot.get_memory("u"), ["Project: agents"])

    def test_users_are_isolated(self) -> None:
        bot = self.bot()
        bot.chat("a", "u", "My name is Ann.")
        self.assertEqual(bot.get_memory("stranger"), [])
        reply = bot.chat("b", "stranger", "Do you remember my name?")
        self.assertNotIn("Ann", reply["answer"])

    def test_recall_crosses_sessions_without_citations(self) -> None:
        bot = self.bot()
        bot.chat("a", "u", "My name is Ann.")
        reply = bot.chat("b", "u", "What is my name?")
        self.assertIn("Ann", reply["answer"])
        self.assertEqual(reply["citations"], [])

    def test_paper_questions_are_not_recall(self) -> None:
        bot = self.bot()
        for question in ("What is my best option for retrieval?", "Do you know how BERT works?"):
            with self.subTest(question):
                self.assertTrue(bot.chat("a", "u", question)["citations"])

    def test_forget_and_reset(self) -> None:
        bot = self.bot()
        bot.chat("a", "u", "My project is about RAG. I prefer short answers.")
        bot.chat("a", "u", "Forget my project.")
        self.assertEqual(bot.get_memory("u"), ["Preference: short answers"])
        bot.chat("a", "u", "Forget everything about me.")
        self.assertEqual(bot.get_memory("u"), [])

    def test_memory_path_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "memory.db")
            self.bot(memory_path=path).chat("a", "u", "My name is Ann.")
            self.assertEqual(self.bot(memory_path=path).get_memory("u"), ["Name: Ann"])


class Conversation(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedHandler)
        self.server.requests = []
        self.server.rewrite_reply = "Query: Direct Preference Optimization reward model"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def bot(self, **extra) -> TABot:
        return TABot(
            {
                "endpoint": self.endpoint,
                "model": "mock-model",
                "corpus_path": str(ROOT / "corpus"),
                "temperature": 0.0,
                "seed": 42,
                **extra,
            }
        )

    def test_rewrite_label_is_stripped_and_used(self) -> None:
        bot = self.bot()
        bot.chat("s", "u", "How does DPO work?")
        self.assertEqual(bot._rewrite("its limits?", []), "Direct Preference Optimization reward model")
        self.assertIn("L10-04", bot.chat("s", "u", "What are its limitations?")["citations"])

    def test_rewrite_can_be_disabled(self) -> None:
        bot = self.bot(rewrite=False)
        bot.chat("s", "u", "How does DPO work?")
        bot.chat("s", "u", "What are its limitations?")
        self.assertEqual(len(self.server.requests), 2)

    def test_history_stays_inside_user_and_session(self) -> None:
        bot = self.bot()
        bot.chat("s", "u", "How does DPO work?")
        for session, user in (("s", "other"), ("new", "u")):
            self.server.requests.clear()
            bot.chat(session, user, "Hello there")
            self.assertEqual(len(self.server.requests), 1)
            self.assertEqual(len(self.server.requests[0]["messages"]), 2)

    def test_config_reaches_the_endpoint(self) -> None:
        self.bot().chat("s", "u", "What is DPO?")
        request = self.server.requests[-1]
        self.assertEqual(
            (request["model"], request["temperature"], request["seed"]), ("mock-model", 0.0, 42)
        )


class EndpointFailures(BotTestCase):
    def test_unreachable_endpoint_raises_timeout(self) -> None:
        with self.assertRaises(TimeoutError):
            self.bot(endpoint="http://127.0.0.1:9", timeout=1).chat("s", "u", "What is DPO?")


if __name__ == "__main__":
    unittest.main()
