# AI CAN Log Analyzer

A complete toolchain for converting, analyzing, and querying large CAN (Controller Area Network) log datasets using natural language, powered entirely by local AI models.

## Overview

The AI CAN Log Analyzer consists of three main components:

1. **BLF to CSV Converter (`blf_to_csv.py`)**: A robust parser to convert raw Vector BLF (Binary Logging Format) files into time-aligned CSV datasets using DBC files.
2. **Log Analyzer Engine (`analyzer.py`)**: A multiprocessing-enabled analytics engine that uses local LLMs and embeddings to understand natural language queries and map them to physical CAN signals and logical conditions.
3. **Web Interface (`app.py`)**: A modern, interactive Streamlit frontend that provides a ChatGPT-like experience for querying automotive log data.

---

## Components Detail

### 1. Data Conversion (`blf_to_csv.py`)
Vector BLF files are the industry standard for raw CAN logging, but they are difficult to analyze directly. This script:
- Reads all `.blf` files in a designated folder.
- Uses one or more `.dbc` files to decode raw CAN frames into physical signal values.
- Aligns signals temporally, creating a structured CSV file where columns represent specific signals formatted as `<DBC>::<Message>::<Signal>[unit]`.

### 2. AI Analytics Engine (`analyzer.py`)
This is the core intelligence of the application, running entirely locally on CPU. 
- **Retrieval-Augmented Generation (RAG)**: Uses the `all-MiniLM-L6-v2` embedding model to semantically match plain-english signal names mentioned by the user (e.g., "pack current") to the exact, often cryptic DBC signal headers (e.g., `BMS_MSG::Pack_Current[A]`).
- **Query Parsing**: Uses a lightweight instruction-tuned LLM (`Qwen/Qwen2.5-0.5B-Instruct`) to translate user requests (e.g., "Find logs where current is greater than 10A") into strict JSON structures containing mathematical operators and threshold values.
- **Multiprocessing**: Optimized to handle 100+ CSV log files simultaneously using `concurrent.futures.ProcessPoolExecutor`, drastically reducing search and evaluation time across large log repositories.

### 3. User Interface (`app.py`)
A Streamlit web application providing a sleek, dark-themed UI.
- Chat interface for natural language interaction.
- Instant display of AI parsing logic (matched signal, applied condition).
- Tabular display of timestamps when specific conditions occurred across the logs.
- Export functionality to download the filtered results as CSV.

---

## Setup & Usage

### Prerequisites
Make sure you have the required dependencies installed (e.g., `streamlit`, `pandas`, `transformers`, `sentence-transformers`, `torch`, `cantools`, `python-can`).

### Running the App
Start the Streamlit interface by running:
```bash
streamlit run app.py
```
*Note: The first time you run the app, it will download the open-source LLM and Embedding models to a local `./local_models` directory.*

### Workflow
1. Use `blf_to_csv.py` to convert raw CAN logs into CSV files.
2. Ensure the CSV files are placed in the folder monitored by the application (or edit the `folder_path` in `app.py`).
3. Ask questions in the Streamlit UI, such as:
   - *"Show me when the battery voltage was less than 300V"*
   - *"Give me logs where GC_DC_CURR is greater than 7A"*
