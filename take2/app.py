from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from can_log_analyzer import CanLogDataset, result_summary
from local_llm import LocalGGUFLLM, LocalLLMError


@st.cache_resource(show_spinner="Loading local GGUF model…")
def load_local_llm(model_path: str, context_size: int, gpu_layers: int) -> LocalGGUFLLM:
    return LocalGGUFLLM(model_path, context_size=context_size, gpu_layers=gpu_layers)

st.set_page_config(page_title="Local CAN Log RAG", page_icon="🚗", layout="wide")
st.title("Local CAN Log RAG Analyzer")
st.caption("CSV, local GGUF LLM inference, signal retrieval, and threshold calculations stay on this computer. No Ollama or cloud API is used.")

with st.sidebar:
    st.header("Load CAN CSV")
    uploaded = st.file_uploader("CSV log", type=["csv"])
    st.markdown("Expected layouts: wide, long, or CANape `Time[s], signal, blank` triplets.")
    st.divider()
    st.header("Local LLM (required)")
    model_path = st.text_input("GGUF instruct-model path", value=os.getenv("CAN_LOG_LLM_PATH", ""), placeholder=r"D:\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    context_size = st.number_input("Context size", min_value=1024, max_value=32768, value=4096, step=1024)
    gpu_layers = st.number_input("GPU layers (0 = CPU)", min_value=0, max_value=200, value=0, step=1)
    st.caption("Uses llama-cpp-python to run a GGUF model on this machine.")
    st.divider()
    st.markdown("**Queries to try**")
    st.code("GC_DC_CURR_PERC_IN greater than 7A for more than 2 seconds", language=None)
    st.code("pack_current signal value exceeds 80A", language=None)

if not uploaded:
    st.info("Upload a CAN log CSV to begin.")
    st.stop()

try:
    payload = uploaded.getvalue()
    dataset = CanLogDataset.from_csv(io.BytesIO(payload), uploaded.name)
except Exception as error:
    st.error(f"Unable to read this CSV: {error}")
    st.stop()

left, right = st.columns([2, 1])
with left:
    question = st.text_input("Ask about the CAN log", placeholder="e.g. pack_current exceeds 80A")
with right:
    st.metric("Log samples", f"{sum(len(values) for values in dataset._signal_frames.values()):,}")
    st.metric("Numeric signals", f"{len(dataset.signals):,}")

with st.expander("Detected signal catalog"):
    st.dataframe(pd.DataFrame({"signal": dataset.signals}), hide_index=True, use_container_width=True)

if question:
    if not model_path.strip():
        st.error("Set the path to a local GGUF instruct model to run a natural-language query.")
        st.stop()
    try:
        llm = load_local_llm(model_path.strip(), int(context_size), int(gpu_layers))
        condition = llm.plan(question, dataset)
        result = dataset.execute(condition)
        answer = llm.answer(question, result)
        st.subheader("Local LLM answer")
        st.write(answer)
        st.success(result_summary(result))
        st.caption(f"The local LLM selected `{result['matched_signal']}` from retrieved candidates. Numeric evaluation and duration verification were deterministic.")
        st.subheader("Native signal cycle time")
        st.dataframe(pd.DataFrame([result["cycle_time"]]), hide_index=True, use_container_width=True)
        if result["signal_suggestions"]:
            st.dataframe(pd.DataFrame(result["signal_suggestions"]), hide_index=True, use_container_width=True)
        periods = pd.DataFrame(result["matching_periods"])
        if not periods.empty:
            st.subheader("Matching periods")
            st.dataframe(periods, hide_index=True, use_container_width=True)
            st.subheader("Matching log rows")
            rows = pd.DataFrame(result["rows"])
            st.dataframe(rows, hide_index=True, use_container_width=True, height=320)
            st.download_button("Download matching rows as CSV", rows.to_csv(index=False).encode("utf-8"), "matching_can_logs.csv", "text/csv")
        with st.expander("Query plan (auditable)"):
            st.json({"parsed_condition": result["condition"], "matched_signal": result["matched_signal"], "source": uploaded.name})
    except LocalLLMError as error:
        st.error(f"Local LLM error: {error}")
    except Exception as error:
        st.error(str(error))
