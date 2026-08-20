"""
Ingest CAN log CSV files into a SQLite database.

Supports two common CSV shapes automatically:
  1. LONG format: columns like Timestamp, Signal, Value  (one row per reading)
  2. WIDE format: columns like Timestamp, Signal_A, Signal_B, Signal_C, ...
     (one row per time sample, each column a signal)

Normalizes everything into a single long table:
    readings(log_file TEXT, timestamp REAL, signal TEXT, value REAL)

Usage:
    python ingest.py --logs_dir ./logs --db can_logs.db
"""

import argparse
import glob
import os
import sqlite3
import sys

import pandas as pd

TIME_COL_CANDIDATES = ["timestamp", "time", "time_s", "time (s)", "abs_time"]


def find_time_column(columns):
    lower = {c.lower().strip(): c for c in columns}
    for cand in TIME_COL_CANDIDATES:
        if cand in lower:
            return lower[cand]
    # fallback: first column
    return columns[0]


def is_long_format(columns):
    lower = [c.lower().strip() for c in columns]
    has_signal_col = any(c in ("signal", "signal_name", "channel", "name") for c in lower)
    has_value_col = any(c in ("value", "val", "data") for c in lower)
    return has_signal_col and has_value_col


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_long(df, log_file):
    lower_map = {c.lower().strip(): c for c in df.columns}
    time_col = find_time_column(df.columns)
    signal_col = lower_map.get("signal") or lower_map.get("signal_name") or lower_map.get("channel") or lower_map.get("name")
    value_col = lower_map.get("value") or lower_map.get("val") or lower_map.get("data")

    out = pd.DataFrame({
        "log_file": log_file,
        "timestamp": pd.to_numeric(df[time_col], errors="coerce"),
        "signal": df[signal_col].astype(str).str.strip(),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
    })
    return out.dropna(subset=["timestamp", "signal", "value"])


def normalize_wide(df, log_file):
    time_col = find_time_column(df.columns)
    signal_cols = [c for c in df.columns if c != time_col]

    long_df = df.melt(id_vars=[time_col], value_vars=signal_cols,
                       var_name="signal", value_name="value")
    long_df = long_df.rename(columns={time_col: "timestamp"})
    long_df["log_file"] = log_file
    long_df["timestamp"] = pd.to_numeric(long_df["timestamp"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df.dropna(subset=["timestamp", "signal", "value"])[
        ["log_file", "timestamp", "signal", "value"]
    ]


def ingest_file(path, conn):
    log_file = os.path.basename(path)
    df = load_csv(path)

    if is_long_format(df.columns):
        norm = normalize_long(df, log_file)
    else:
        norm = normalize_wide(df, log_file)

    norm.to_sql("readings", conn, if_exists="append", index=False)
    return len(norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs", help="Folder containing CSV log files")
    ap.add_argument("--db", default="can_logs.db", help="Output SQLite database path")
    ap.add_argument("--fresh", action="store_true", help="Drop existing table before ingest")
    args = ap.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.logs_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {args.logs_dir}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    if args.fresh:
        conn.execute("DROP TABLE IF EXISTS readings")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            log_file TEXT,
            timestamp REAL,
            signal TEXT,
            value REAL
        )
    """)

    total = 0
    for path in csv_files:
        try:
            n = ingest_file(path, conn)
            total += n
            print(f"  {os.path.basename(path)}: {n} readings")
        except Exception as e:
            print(f"  FAILED {os.path.basename(path)}: {e}", file=sys.stderr)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_signal ON readings(signal)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_log_file ON readings(log_file)")
    conn.commit()
    conn.close()

    print(f"\nDone. Ingested {total} readings from {len(csv_files)} files into {args.db}")


if __name__ == "__main__":
    main()
