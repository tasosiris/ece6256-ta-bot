"""Rough local answer scorer, kept out of the submission.

This is NOT the course judge: it applies only the four criteria the README names.
Usage: python tests/judge_check.py ENDPOINT MODEL answers.json > scores.json
"""

import json
import re
import sys

import requests

PROMPT = """You are grading a course teaching assistant's answer.
Score each criterion from 1 (poor) to 5 (excellent):
accuracy, relevance, completeness, grounding.
Reply with one JSON object using those four keys and a short "reason".

Question: {q}

Answer: {answer}"""

endpoint, model, path = sys.argv[1:4]
scores = []
for item in json.load(open(path)):
    response = requests.post(endpoint.rstrip("/") + "/chat/completions", timeout=120, json={
        "model": model, "temperature": 0, "seed": 42, "max_tokens": 200,
        "messages": [{"role": "user", "content": PROMPT.format(**item)}]})
    text = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.S)
    try:
        parsed = json.loads(match.group(0))
    except (AttributeError, ValueError):
        parsed = {"raw": text}
    parsed["id"] = item["id"]
    scores.append(parsed)
print(json.dumps(scores, indent=1))
