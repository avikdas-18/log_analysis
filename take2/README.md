# Local CAN Log RAG Analyzer

A local Streamlit application for asking natural-language questions about CAN signal CSV logs. It runs a GGUF instruct model via `llama-cpp-python`; it does **not** use Ollama or make network calls.

## What it does

- Imports wide CSV logs: `timestamp,GC_DC_CURR_PERC_IN,pack_current,...`
- Imports long CSV logs: `timestamp,signal,value`
- Imports CANape-style exports with repeated `Time[s], signal, blank` triplets (the supplied sample format)
- Uses a local TF-IDF vector index over the detected signal catalog to retrieve candidates (the retrieval part of RAG)
- Gives the user question and retrieved candidates to the local GGUF LLM, which creates a constrained JSON query plan
- Validates that plan, executes the numeric and continuous-duration test deterministically, then gives the verified evidence to the same local LLM for its answer

The LLM is in the loop for both natural-language interpretation and response generation, but it never decides the math. This keeps results auditable and prevents LLM hallucinations from selecting logs.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Download or copy a GGUF instruct model to your computer, then enter its local path in the sidebar (or set `CAN_LOG_LLM_PATH`). Open the local URL printed by Streamlit, upload `sample_data.csv`, then ask a question.

Or use the dependency-light command-line interface:

```powershell
py cli.py .\sample_data.csv "pack_current signal value exceeds 80A" --model D:\models\your-instruct-model.gguf --output .\matching_can_logs.csv
```

## Local model requirement (no Ollama)

Use any compatible local **GGUF instruct model** with `llama-cpp-python`; a 3B–8B quantized model is usually sufficient for query planning. The app does not download a model or transmit log data. Keep the deterministic `CanLogDataset.execute()` stage unchanged so actual log selection remains verifiable.

## Assumptions for duration

Each signal has its own timestamp stream and its own nominal cycle time (median positive interval). For a request like "above 7A for more than 2 seconds", samples must be consecutive and timestamps must not have a gap greater than 1.5× **that signal's** nominal cycle. A value is held until the next on-cycle timestamp only; at a dropped-message gap it is held for at most one nominal cycle. This prevents falsely bridging gaps between asynchronous CAN messages.
