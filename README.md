# CAN Log AI Query Tool

Local, Ollama-powered natural language search over CAN bus log CSVs.

## How it works

1. **Ingest**: all your CSV logs get parsed into a single SQLite database of
   `(log_file, timestamp, signal, value)` readings. Numeric filtering always
   happens in SQL, never by asking the LLM to eyeball a table — this is what
   makes results reliable.
2. **Rules (`rules.yaml`)**: define your domain concepts once, e.g. "generator
   operational" = `GEN_CURRENT > 5 AND GEN_VOLTAGE > 200`. Use your team's real
   signal names and thresholds here.
3. **Query engine**: your natural language question first gets matched against
   known rule concepts by the LLM. If nothing matches, the LLM falls back to
   extracting a signal/operator/threshold directly from your question.
4. **Ollama** only ever does language tasks (matching intent, summarizing
   results) — never numeric comparison over raw data.

## Setup

```bash
# 1. Install Ollama and pull a model
#    https://ollama.com/download
ollama pull llama3.1

# 2. Python deps
pip install pandas pyyaml requests

# 3. Put your CSV logs in ./logs/

# 4. Ingest logs into the database
python ingest.py --logs_dir ./logs --db can_logs.db --fresh

# 5. Edit rules.yaml with your real signal names + thresholds

# 6. Run the query tool
python main.py
```

## Example session

```
Query> show me the logs where generator is operational
[matched rule: 'generator operational']
Condition used: {'logic': 'AND', 'conditions': [...]}

  log_2024_03_01.csv  (142 matching readings)
      t=12.40  GEN_CURRENT=6.2
      t=12.40  GEN_VOLTAGE=214.0
      ...

Summary: The generator was operational in log_2024_03_01.csv, with current
and voltage readings above threshold across a 45-second window starting at
t=12.4s.
```

## CSV format support

`ingest.py` auto-detects two shapes:

- **Long**: columns like `Timestamp, Signal, Value`
- **Wide**: columns like `Timestamp, GEN_CURRENT, GEN_VOLTAGE, BATT_SOC, ...`

If your logs use different column names, adjust `TIME_COL_CANDIDATES` and the
column-detection logic in `ingest.py`.

## Notes / next steps

- The "AND" condition in `_build_sql` currently checks per-row matches, not
  per-timestamp joins across signals. For strict "current AND voltage above
  threshold at the *same instant*" logic, you'll want to bucket readings by a
  rounded timestamp and pivot before filtering — ask if you want this added,
  it's a straightforward extension.
- Swap `MODEL` in `query_engine.py` for any Ollama model you prefer
  (e.g. `mistral`, `qwen2.5`) — smaller models work fine for this
  classification/extraction task.
- For very large log databases, consider DuckDB instead of SQLite — same
  ingest logic, faster on wide analytical queries.
