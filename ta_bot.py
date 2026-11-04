"""
CS 6604 / ECE 6256 — Assignment 3: The Course TA-Bot
Conversational RAG with Memory

Starter stub. Implement the methods below.
Do NOT rename the class or change the method signatures.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import requests

CANDIDATE_CHUNKS = 40  # chunks pulled from BM25 before grouping by paper
TOP_PAPERS = 3  # papers cited per turn; hit-rate rewards recall over precision
CHUNKS_PER_PAPER = 2  # evidence passages kept for each cited paper
TITLE_WEIGHT = 5.0  # papers are usually asked about by what their title names
MAX_ANSWER_TOKENS = 500
MAX_REWRITE_TOKENS = 64
MAX_REWRITE_CHARS = 300
HISTORY_TURNS = 3  # earlier exchanges kept for rewriting and generation
RRF_K = 60  # standard reciprocal-rank-fusion damping constant
DEFAULT_TIMEOUT_S = 60.0

ANSWER_SYSTEM_PROMPT = """You are the teaching assistant for a graduate course on \
conversational AI. Answer the student's question using only the numbered passages from \
the course papers.

Rules:
- Base every claim on the passages. Do not add outside knowledge.
- Name the paper you are drawing on, using its title.
- If the passages do not contain the answer, say so plainly instead of guessing.
- Be accurate and complete, and keep the answer focused on the question."""

STOPWORDS = frozenset(
    """a about across all also an and any are as at be been but by can could did do
    does explain describe for from had has have how i in into is it its me my of on
    or our paper papers please should so such summarize tell than that the their them
    then these they this those to us used using was we were what when where which
    while who why will with would you your""".split()
)
REWRITE_SYSTEM_PROMPT = """Rewrite the student's latest message as one standalone search \
query for finding course papers. Replace pronouns and references such as "it" or "that \
approach" with what they refer to in the conversation. Output only the query."""

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_REFERS_BACK_RE = re.compile(
    r"\b(?:these|those|both|so far|earlier|previous(?:ly)?|above|mentioned"
    r"|all (?:two|three|four|of (?:them|these|those))|the (?:two|three|four)"
    r"|we (?:discussed|talked|covered)"
    r"|(?:that|this) (?:approach|method|model|paper|technique|system))\b",
    re.IGNORECASE,
)


def _fts_query(text: str) -> str:
    """Build an FTS5 OR query; quoting stops user text being parsed as FTS syntax."""
    terms = dict.fromkeys(
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS and len(token) > 1
    )
    return " OR ".join(f'"{term}"' for term in terms)


class TABot:
    """Course TA-Bot with RAG retrieval and cross-session memory."""

    def __init__(self, config: dict) -> None:
        """
        Initialise the TA-Bot.

        Parameters
        ----------
        config : dict
            Required keys set by the harness (do not hard-code their values):
                endpoint   : str   – vLLM-compatible base URL
                model      : str   – model name served at the endpoint
                corpus_path: str   – path to the paper corpus directory
                temperature: float – generation temperature (0 for graded runs)
                seed       : int   – random seed for reproducibility
            You may add custom keys for your own use.
        """
        self.config = config
        db_path = Path(config["corpus_path"]) / "corpus.db"
        self._corpus = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        # The mock's URL has no /v1 suffix but real vLLM URLs usually do.
        self._base_url = config["endpoint"].rstrip("/").removesuffix("/v1")
        self._timeout = config.get("timeout", DEFAULT_TIMEOUT_S)
        self._rewrite_enabled = config.get("rewrite", True)  # ablation switch
        self._sessions: dict[tuple[str, str], list[dict]] = {}

    def _complete(self, messages: list[dict], max_tokens: int = MAX_ANSWER_TOKENS) -> str:
        """Call the chat endpoint; any endpoint failure surfaces as TimeoutError."""
        try:
            response = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self.config["model"],
                    "messages": messages,
                    "temperature": self.config["temperature"],
                    "seed": self.config["seed"],
                    "max_tokens": max_tokens,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            raise TimeoutError(f"Endpoint request failed: {exc}") from exc
        return (content or "").strip()

    @staticmethod
    def _format_passages(papers: list[dict]) -> str:
        """Render retrieved chunks as numbered, attributed evidence."""
        passages = []
        for paper in papers:
            for chunk in paper["chunks"]:
                number = len(passages) + 1
                header = f'[{number}] "{paper["title"]}" ({paper["paper_id"]})'
                if chunk["section"]:
                    header += f', section: {chunk["section"]}'
                passages.append(f'{header}\n{chunk["text"]}')
        return "\n\n".join(passages)

    def _retrieve(self, query: str) -> list[dict]:
        """Return the top papers for a query, each with its best chunks."""
        match = _fts_query(query)
        if not match:
            return []
        rows = self._corpus.execute(
            """SELECT c.chunk_id, c.paper_id, p.title, c.section, c.text,
                      bm25(chunks_fts, 0, 0, ?, 1.0, 1.0) AS score
               FROM chunks_fts
               JOIN chunks c USING (chunk_id)
               JOIN papers p USING (paper_id)
               WHERE chunks_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (TITLE_WEIGHT, match, CANDIDATE_CHUNKS),
        ).fetchall()

        papers: dict[str, dict] = {}
        for _, paper_id, title, section, text, score in rows:
            paper = papers.setdefault(
                paper_id, {"paper_id": paper_id, "title": title, "score": 0.0, "chunks": []}
            )
            if len(paper["chunks"]) < CHUNKS_PER_PAPER:
                # FTS5 bm25() is negative; flip it so that bigger means better.
                paper["score"] += -score
                paper["chunks"].append({"section": section, "text": text})
        ranked = sorted(papers.values(), key=lambda p: (-p["score"], p["paper_id"]))
        return ranked[:TOP_PAPERS]

    def _rewrite(self, message: str, history: list[dict]) -> str | None:
        """Turn a follow-up into a standalone query; None if the reply is unusable."""
        transcript = "\n".join(
            f'Student: {turn["question"]}\nTA: {turn["answer"]}' for turn in history
        )
        reply = self._complete(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation:\n{transcript}\n\nLatest message: {message}",
                },
            ],
            max_tokens=MAX_REWRITE_TOKENS,
        )
        query = reply.split("\n", 1)[0].strip(" \"'")
        return query if query and len(query) <= MAX_REWRITE_CHARS else None

    def _gather_papers(self, message: str, history: list[dict]) -> list[dict]:
        """Retrieve for this turn, then add papers the student is pointing back at."""
        papers = self._retrieve(message)
        if history and self._rewrite_enabled:
            rewritten = self._rewrite(message, history)
            if rewritten:
                # Fuse rather than replace: a bad rewrite must not lose the original hits.
                scores: dict[str, float] = {}
                by_id: dict[str, dict] = {}
                for ranking in (papers, self._retrieve(rewritten)):
                    for rank, paper in enumerate(ranking):
                        by_id.setdefault(paper["paper_id"], paper)
                        scores[paper["paper_id"]] = (
                            scores.get(paper["paper_id"], 0.0) + 1 / (RRF_K + rank)
                        )
                ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))
                papers = [by_id[pid] for pid in ordered[:TOP_PAPERS]]

        if _REFERS_BACK_RE.search(message):
            seen = {paper["paper_id"] for paper in papers}
            for turn in history:
                primary = turn["primary"]
                if primary and primary["paper_id"] not in seen:
                    seen.add(primary["paper_id"])
                    papers.append(primary)
        return papers

    def chat(
        self,
        session_id: str,
        user_id: str,
        message: str,
    ) -> dict:
        """
        Process one user turn and return an answer with citations.

        Parameters
        ----------
        session_id : str
            Conversation scope — a new session_id starts a fresh
            conversation but keeps the user's cross-session memory.
        user_id : str
            Memory scope — identifies the user whose memory store
            to read from and write to.
        message : str
            The user's message.

        Returns
        -------
        dict
            {"answer": str, "citations": list[str]}
            - answer: the bot's response (complete, no streaming)
            - citations: list of paper_id strings from the corpus
              that were used to produce the answer. May be empty
              if no retrieval was needed.

        Raises
        ------
        TimeoutError
            You MUST raise TimeoutError if the endpoint is
            unreachable or a request exceeds your timeout.
            The harness catches it and marks this turn as 0.
            Do NOT let the process hang — raise, don't wait.
        """
        # Keyed by user too, so a reused session_id can never cross between users.
        session = self._sessions.setdefault((user_id, session_id), [])
        history = session[-HISTORY_TURNS:]
        papers = self._gather_papers(message, history)
        evidence = self._format_passages(papers) or "(no passages found)"

        messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append(
            {"role": "user", "content": f"Passages:\n{evidence}\n\nQuestion: {message}"}
        )
        answer = self._complete(messages)

        session.append(
            {
                "question": message,
                "answer": answer,
                "primary": papers[0] if papers else None,
            }
        )
        return {"answer": answer, "citations": [p["paper_id"] for p in papers]}

    def reset_memory(self, user_id: str) -> None:
        """
        Clear all stored memory for the given user.

        The harness calls this before each memory-probe sequence
        to ensure a clean slate. After this call, ``get_memory(user_id)``
        must return an empty list.

        Parameters
        ----------
        user_id : str
            The user whose memory to clear.
        """
        # TODO: clear the memory store for this user
        raise NotImplementedError("Implement reset_memory")

    def get_memory(self, user_id: str) -> list[str]:
        """
        Return the current memory store for the given user as a
        list of human-readable fact strings.

        This is read-only — calling it must not change the store.
        The harness uses it to inspect what your bot chose to remember
        (and what it chose not to remember).

        Any internal representation is fine as long as each entry
        serialises to a readable string.

        Parameters
        ----------
        user_id : str
            The user whose memory to inspect.

        Returns
        -------
        list[str]
            Stored facts as readable strings. Empty list if nothing
            is stored (or after ``reset_memory``).
        """
        # TODO: return the contents of the memory store
        raise NotImplementedError("Implement get_memory")
