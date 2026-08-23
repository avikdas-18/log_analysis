"""Local RAG-assisted query engine for CAN signal CSVs.

The numeric filter is deliberately deterministic: an LLM may identify a signal,
but it never decides which samples satisfy a threshold.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TIME_ALIASES = {"time", "timestamp", "time_s", "time_sec", "seconds", "ts", "datetime", "date_time"}
SIGNAL_ALIASES = {"signal", "signal_name", "name", "parameter", "channel", "can_signal"}
VALUE_ALIASES = {"value", "signal_value", "raw_value", "data", "val"}


@dataclass(frozen=True)
class Condition:
    signal: str
    operator: str
    threshold: float
    duration_seconds: float | None = None
    duration_operator: str = ">="


def _normalise(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _find_column(columns: Iterable[object], aliases: set[str]) -> str | None:
    for column in columns:
        if _normalise(column) in aliases:
            return str(column)
    return None


def _coerce_time(values: pd.Series) -> pd.Series:
    """Return elapsed seconds, supporting numeric or datetime CAN timestamps."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return numeric.astype(float)
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    if timestamps.notna().any():
        return (timestamps - timestamps.dropna().iloc[0]).dt.total_seconds()
    raise ValueError("The timestamp column is neither numeric seconds nor a readable date/time.")


class SignalCatalog:
    """A lightweight local vector index over signal names and readable aliases."""

    def __init__(self, signals: Iterable[str]):
        self.signals = list(dict.fromkeys(map(str, signals)))
        if not self.signals:
            raise ValueError("No numeric CAN signals were found in the CSV.")
        documents = [self._document(signal) for signal in self.signals]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True)
        self.matrix = self.vectorizer.fit_transform(documents)

    @staticmethod
    def _document(signal: str) -> str:
        readable = re.sub(r"[_\\-]+", " ", signal)
        return f"{signal} {readable} {readable.replace('curr', 'current')}"

    def search(self, wording: str, limit: int = 5) -> list[tuple[str, float]]:
        needle = self.vectorizer.transform([self._document(wording)])
        scores = cosine_similarity(needle, self.matrix).ravel()
        indices = np.argsort(scores)[::-1][:limit]
        return [(self.signals[index], float(scores[index])) for index in indices]


class CanLogDataset:
    """Normalises wide, long, and CANape-style multi-timebase CAN CSVs."""

    def __init__(self, frame: pd.DataFrame, source_name: str = "uploaded CSV"):
        self.source_name = source_name
        self.frame, self.time_column = self._normalise_input(frame)
        self.signals = [column for column in self.frame.columns if column != self.time_column and pd.api.types.is_numeric_dtype(self.frame[column])]
        self._signal_frames = {
            signal: self.frame[[self.time_column, signal]].rename(columns={self.time_column: "time_s", signal: "value"}).dropna(subset=["time_s", "value"]).reset_index(names="source_row")
            for signal in self.signals
        }
        self.catalog = SignalCatalog(self.signals)

    @classmethod
    def from_csv(cls, path_or_buffer: Any, source_name: str = "uploaded CSV") -> "CanLogDataset":
        try:
            frame = pd.read_csv(path_or_buffer, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            # CANape / Windows exports commonly carry degree symbols in cp1252.
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            frame = pd.read_csv(path_or_buffer, low_memory=False, encoding="cp1252")
        canape = cls._canape_signal_frames(frame)
        if canape:
            instance = cls.__new__(cls)
            instance.source_name = source_name
            instance.frame = pd.DataFrame({"__time_seconds__": []})
            instance.time_column = "__time_seconds__"
            instance._signal_frames = canape
            instance.signals = list(canape)
            instance.catalog = SignalCatalog(instance.signals)
            return instance
        return cls(frame, source_name)

    @staticmethod
    def _canape_signal_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Decode repeating `Time[s], Signal, blank` CANape export triplets."""
        columns = list(frame.columns)
        time_indices = [
            index for index, column in enumerate(columns)
            if _normalise(column) in TIME_ALIASES or str(column).strip().lower().startswith("time[s]")
        ]
        # A normal wide CSV has one shared timestamp. CANape has many independent
        # timestamp columns and needs the special per-signal handling below.
        if len(time_indices) < 2:
            return {}
        parsed: dict[str, pd.DataFrame] = {}
        for index in time_indices:
            if index >= len(columns) - 1:
                continue
            # `read_csv` makes duplicate Time[s] headers unique with `.1`, `.2`, …
            # so test the original-looking prefix as well as normalised aliases.
            raw_time_name = str(columns[index]).strip().lower()
            if _normalise(columns[index]) not in TIME_ALIASES and not raw_time_name.startswith("time[s]"):
                continue
            signal = str(columns[index + 1])
            if not signal or signal.lower().startswith("unnamed"):
                continue
            values = pd.DataFrame({
                "time_s": pd.to_numeric(frame.iloc[:, index], errors="coerce"),
                "value": pd.to_numeric(frame.iloc[:, index + 1], errors="coerce"),
            }).dropna(subset=["time_s", "value"])
            if not values.empty:
                parsed[signal] = values.sort_values("time_s").reset_index(names="source_row")
        return parsed

    @staticmethod
    def _normalise_input(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        frame = frame.copy()
        if frame.empty:
            raise ValueError("The CSV contains no log rows.")
        time_input = _find_column(frame.columns, TIME_ALIASES)
        if not time_input:
            raise ValueError("Could not find a timestamp column. Use time, timestamp, time_s, or seconds.")
        signal_column = _find_column(frame.columns, SIGNAL_ALIASES)
        value_column = _find_column(frame.columns, VALUE_ALIASES)
        if signal_column and value_column:  # long format
            frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
            frame["__time_seconds__"] = _coerce_time(frame[time_input])
            wide = frame.pivot_table(index="__time_seconds__", columns=signal_column, values=value_column, aggfunc="last").reset_index()
            wide.columns.name = None
            return wide.sort_values("__time_seconds__").reset_index(drop=True), "__time_seconds__"
        frame["__time_seconds__"] = _coerce_time(frame[time_input])
        for column in frame.columns:
            if column not in {time_input, "__time_seconds__"}:
                frame[column] = pd.to_numeric(frame[column], errors="ignore")
        # The original timestamp is metadata, not a queryable numeric CAN signal.
        frame = frame.drop(columns=[time_input])
        return frame.sort_values("__time_seconds__").reset_index(drop=True), "__time_seconds__"

    def signal_suggestions(self, phrase: str) -> list[dict[str, Any]]:
        return [{"signal": signal, "score": round(score, 3)} for signal, score in self.catalog.search(phrase)]

    def cycle_time(self, signal: str) -> dict[str, float | int | None]:
        """Calculate timing statistics from this signal's own CAN timestamps."""
        times = self._signal_frames[signal]["time_s"].to_numpy(dtype=float)
        intervals = np.diff(times)
        intervals = intervals[intervals > 0]
        if len(intervals) == 0:
            return {"samples": int(len(times)), "nominal_cycle_s": None, "min_cycle_s": None, "max_cycle_s": None, "gap_limit_s": None}
        nominal = float(np.median(intervals))
        # Long gaps indicate dropped / absent messages, never a held signal value.
        gap_limit = nominal * 1.5
        return {
            "samples": int(len(times)), "nominal_cycle_s": round(nominal, 9),
            "min_cycle_s": round(float(intervals.min()), 9), "max_cycle_s": round(float(intervals.max()), 9),
            "gap_limit_s": round(gap_limit, 9),
        }

    def resolve_signal(self, requested: str) -> tuple[str, list[dict[str, Any]]]:
        canonical = _normalise(requested)
        exact = [signal for signal in self.signals if _normalise(signal) == canonical]
        if exact:
            return exact[0], self.signal_suggestions(requested)
        choices = self.signal_suggestions(requested)
        if not choices or choices[0]["score"] < 0.18:
            raise ValueError(f"Could not match '{requested}' to a signal. Try one of: {', '.join(self.signals[:20])}")
        return choices[0]["signal"], choices

    def execute(self, condition: Condition) -> dict[str, Any]:
        signal, alternatives = self.resolve_signal(condition.signal)
        signal_data = self._signal_frames[signal].copy()
        cycle_time = self.cycle_time(signal)
        values = signal_data["value"]
        if condition.operator == ">":
            matching = values > condition.threshold
        elif condition.operator == ">=":
            matching = values >= condition.threshold
        elif condition.operator == "<":
            matching = values < condition.threshold
        elif condition.operator == "<=":
            matching = values <= condition.threshold
        elif condition.operator == "=":
            matching = values == condition.threshold
        else:
            raise ValueError(f"Unsupported comparison operator: {condition.operator}")
        segments = self._segments(signal_data, matching.fillna(False), condition.duration_seconds, condition.duration_operator, cycle_time["nominal_cycle_s"])
        selected = self._rows_for_segments(signal_data, segments)
        result = {
            "condition": asdict(condition),
            "matched_signal": signal,
            "signal_suggestions": alternatives,
            "cycle_time": cycle_time,
            "matching_rows": int(matching.sum()),
            "matching_periods": segments,
            "rows": selected,
        }
        return result

    def _segments(self, signal_data: pd.DataFrame, matching: pd.Series, minimum_duration: float | None, duration_operator: str = ">=", nominal_cycle: float | None = None) -> list[dict[str, Any]]:
        times = signal_data["time_s"].to_numpy(dtype=float)
        indices = np.flatnonzero(matching.to_numpy())
        if len(indices) == 0:
            return []
        # A missing sample ends a run. For irregular logs, a gap exceeding 1.5x
        # the median sample interval also ends it rather than inventing continuity.
        positive_diffs = np.diff(times)
        baseline = nominal_cycle if nominal_cycle is not None else (float(np.median(positive_diffs[positive_diffs > 0])) if np.any(positive_diffs > 0) else np.inf)
        gap_limit = baseline * 1.5 if np.isfinite(baseline) else np.inf
        groups: list[list[int]] = [[int(indices[0])]]
        for index in indices[1:]:
            if index == groups[-1][-1] + 1 and times[index] - times[groups[-1][-1]] <= gap_limit:
                groups[-1].append(int(index))
            else:
                groups.append([int(index)])
        periods = []
        for group in groups:
            start, end = group[0], group[-1]
            # Treat a value as held only until the next *on-cycle* timestamp.
            # At a dropped-message gap, cap coverage at one native signal cycle.
            next_is_on_cycle = end + 1 < len(times) and times[end + 1] - times[end] <= gap_limit
            duration_end = times[end + 1] if next_is_on_cycle else times[end] + (baseline if np.isfinite(baseline) else 0)
            duration = float(duration_end - times[start])
            meets_duration = minimum_duration is None or (duration > minimum_duration if duration_operator == ">" else duration >= minimum_duration)
            if meets_duration:
                periods.append({
                    "start_time_s": float(times[start]), "end_time_s": float(duration_end),
                    "duration_s": round(duration, 6), "start_row": start, "end_row": end,
                    "samples": len(group),
                })
        return periods

    def _rows_for_segments(self, signal_data: pd.DataFrame, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not segments:
            return []
        row_ids = np.concatenate([np.arange(item["start_row"], item["end_row"] + 1) for item in segments])
        output = signal_data.iloc[row_ids].copy()
        return json.loads(output.to_json(orient="records", date_format="iso"))


def parse_question(question: str, dataset: CanLogDataset) -> Condition:
    """Parse common threshold questions locally; accepts signal names with underscores."""
    clean = question.strip().replace("\\\\", "_").replace("\\", "_")
    duration_match = re.search(r"(?:for\s+)?(?P<qualifier>more than|at least)?\s*(?P<seconds>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", clean, re.I)
    duration = float(duration_match.group("seconds")) if duration_match else None
    duration_operator = ">" if duration_match and (duration_match.group("qualifier") or "").lower() == "more than" else ">="
    comparison = re.search(r"(?P<signal>[A-Za-z][A-Za-z0-9_ .\-]*?)\s+(?:signal(?:\s+value)?\s*)?(?P<op>greater than or equal to|at least|exceeds|greater than|above|over|less than or equal to|at most|below|under|less than|equals?|is)\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?:A|amp(?:ere)?s?)?\b", clean, re.I)
    if not comparison:
        raise ValueError("I could not parse a signal, comparison, and numeric threshold. Example: pack_current exceeds 80A")
    operator_text = comparison.group("op").lower()
    operator = ">" if operator_text in {"exceeds", "greater than", "above", "over", "is"} else "<"
    if operator_text in {"greater than or equal to", "at least"}:
        operator = ">="
    elif operator_text in {"less than or equal to", "at most"}:
        operator = "<="
    elif operator_text in {"equals", "equal"}:
        operator = "="
    requested = re.sub(r"\b(?:give me the logs where|show(?: me)?|find|logs? where|the)\b", "", comparison.group("signal"), flags=re.I).strip(" _.-")
    return Condition(requested, operator, float(comparison.group("value")), duration, duration_operator)


def result_summary(result: dict[str, Any]) -> str:
    condition = result["condition"]
    duration = condition["duration_seconds"]
    duration_text = f" continuously for {'more than' if condition['duration_operator'] == '>' else 'at least'} {duration:g} s" if duration is not None else ""
    periods = result["matching_periods"]
    if not periods:
        return f"No periods found where {result['matched_signal']} {condition['operator']} {condition['threshold']:g}{duration_text}."
    return f"Found {len(periods)} matching period(s) and {sum(period['samples'] for period in periods)} returned log row(s): {result['matched_signal']} {condition['operator']} {condition['threshold']:g}{duration_text}."
