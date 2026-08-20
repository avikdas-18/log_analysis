"""
Core query engine: NL query -> (rule match OR LLM-parsed condition) -> SQL -> results.

Requires a local Ollama server running (default http://localhost:11434)
with a model pulled, e.g.:
    ollama pull llama3.1

Usage as a library:
    from query_engine import QueryEngine
    qe = QueryEngine(db_path="can_logs.db", rules_path="rules.yaml")
    result = qe.query("show me the logs where generator is operational")
"""

import json
import sqlite3

import requests
import yaml

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1"  # change to whatever model you've pulled in ollama


class QueryEngine:
    def __init__(self, db_path="can_logs.db", rules_path="rules.yaml", model=MODEL):
        self.db_path = db_path
        self.model = model
        with open(rules_path) as f:
            self.rules = yaml.safe_load(f)["concepts"]
        self.known_signals = self._load_signal_names()

    def _load_signal_names(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT DISTINCT signal FROM readings").fetchall()
        conn.close()
        return sorted(r[0] for r in rows)

    # ---------- Step 1: try to match a known domain concept ----------
    def _match_concept(self, user_query):
        concept_list = "\n".join(
            f"- {c['name']}: {c['description']}" for c in self.rules
        )
        prompt = f"""You are matching a user question to a known list of domain concepts.

Known concepts:
{concept_list}

User question: "{user_query}"

If the question clearly refers to one of these concepts, respond with ONLY the exact
concept name from the list. If none apply, respond with ONLY the word: NONE

Respond with nothing else."""

        answer = self._call_ollama(prompt).strip()
        for c in self.rules:
            if c["name"].lower() in answer.lower():
                return c
        return None

    # ---------- Step 2: fallback — extract condition directly from query ----------
    def _extract_condition(self, user_query):
        signals_list = ", ".join(self.known_signals)
        prompt = f"""You convert a user question about CAN bus signal logs into a JSON filter.

Available signal names in the database:
{signals_list}

User question: "{user_query}"

Respond with ONLY valid JSON in this exact shape (no markdown, no explanation):
{{
  "logic": "AND",
  "conditions": [
    {{"signal": "<one of the available signal names>", "operator": ">", "threshold": <number>}}
  ]
}}

Pick the operator from: >, <, >=, <=, ==, !=
If the question does not map to a specific numeric threshold, make a reasonable
assumption using the signal's typical operating range."""

        raw = self._call_ollama(prompt).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse LLM output as JSON:\n{raw}")

    def _call_ollama(self, prompt):
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # ---------- Step 3: build SQL and execute ----------
    def _build_sql(self, spec):
        logic = spec.get("logic", "AND").upper()
        clauses = []
        params = []
        for cond in spec["conditions"]:
            clauses.append(f"(signal = ? AND value {cond['operator']} ?)")
            params.extend([cond["signal"], cond["threshold"]])

        joiner = f" {logic} " if logic in ("AND", "OR") else " AND "
        # Each condition is independently matched per-row; to find logs where
        # ALL conditions hold (possibly at different timestamps within the log),
        # we do a per-log_file aggregation.
        sql = f"""
        SELECT log_file, signal, timestamp, value
        FROM readings
        WHERE {joiner.join(clauses)}
        ORDER BY log_file, timestamp
        """
        return sql, params

    def query(self, user_query):
        concept = self._match_concept(user_query)
        if concept:
            spec = {"logic": concept.get("logic", "AND"), "conditions": concept["conditions"]}
            source = f"matched rule: '{concept['name']}'"
        else:
            spec = self._extract_condition(user_query)
            source = "LLM-parsed condition"

        sql, params = self._build_sql(spec)
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        # Group results by log file for readability
        by_log = {}
        for log_file, signal, timestamp, value in rows:
            by_log.setdefault(log_file, []).append(
                {"signal": signal, "timestamp": timestamp, "value": value}
            )

        return {
            "source": source,
            "spec": spec,
            "matching_logs": by_log,
        }

    def summarize(self, user_query, result):
        """Optional: ask the LLM to turn results into a natural language summary."""
        if not result["matching_logs"]:
            return "No logs matched this query."

        log_summaries = []
        for log_file, readings in result["matching_logs"].items():
            times = [r["timestamp"] for r in readings]
            log_summaries.append(
                f"{log_file}: {len(readings)} matching readings, "
                f"time range {min(times):.2f}-{max(times):.2f}"
            )
        facts = "\n".join(log_summaries)

        prompt = f"""User asked: "{user_query}"

Matching data per log file:
{facts}

Write a brief (2-4 sentence) natural language summary of these findings for an
engineer reviewing CAN logs."""

        return self._call_ollama(prompt).strip()
