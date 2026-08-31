import streamlit as st
import pandas as pd
from analyzer import LogAnalyzer
import os

st.set_page_config(page_title="AI CAN Log Analyzer", page_icon="⚡", layout="wide")

# Custom CSS for a more premium look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .css-1d391kg {
        background-color: #1e212b;
    }
    h1, h2, h3 {
        color: #00d2ff;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 210, 255, 0.3);
    }
    .stTextInput>div>div>input {
        background-color: #1e212b;
        color: white;
        border: 1px solid #3a7bd5;
        border-radius: 8px;
    }
    .results-container {
        background: #1e212b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI RAG CAN Log Analyzer")
st.markdown("Analyze large CAN datasets using natural language. Powered by local AI.")

@st.cache_resource
def load_analyzer(data_paths):
    # Load analyzer once and cache it in memory for the given paths
    if data_paths and all(os.path.exists(p) for p in data_paths):
        return LogAnalyzer(list(data_paths))
    return None

# Hardcoded folder path containing CSV logs
folder_path = r"C:\path\to\your\log\folder" # TODO: Update this path to your actual folder

analyzer = None
if folder_path:
    if os.path.isdir(folder_path):
        # Scan for CSV files
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if csv_files:
            st.info(f"📂 Found {len(csv_files)} CSV files in folder.")
            with st.spinner("Initializing AI Model with data. This might take a moment..."):
                analyzer = load_analyzer(tuple(csv_files))
        else:
            st.error("❌ No CSV files found in the specified folder.")
    else:
        st.error("❌ Invalid folder path.")
else:
    default_data = "sample_data.csv"
    if os.path.exists(default_data):
        analyzer = load_analyzer((default_data,))

if analyzer is None:
    st.warning("⚠️ Please provide a valid folder path or ensure `sample_data.csv` exists.")
else:
    if not folder_path:
        st.success(f"✅ Loaded default analyzer with {len(analyzer.signals)} signals.")
    else:
        st.success(f"✅ Loaded analyzer with {len(analyzer.signals)} signals.")
    
    st.markdown("### Ask a Question")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                if "error" in msg:
                    st.error(msg["error"])
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Target Signal", msg["matched_signal"])
                    with col2:
                        st.metric("Condition", msg["condition_str"])
                    with col3:
                        st.metric("Duration Constraint", msg["duration_str"])
                        
                    if msg["results_df"].empty:
                        st.info("No logs matched the specified condition.")
                    else:
                        friendly_cond = msg.get("friendly_condition", msg["condition_str"])
                        st.markdown(f"**Answer**: In the following logs the value of **{msg['matched_signal']}** is **{friendly_cond}**.")
                        st.dataframe(msg["results_df"], use_container_width=True)
                        
                        csv = msg["results_df"].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name='analysis_results.csv',
                            mime='text/csv',
                            key=f"dl_{msg['id']}"
                        )
    
    if query := st.chat_input("Enter your query (e.g. 'Give me the logs where pack_current signal value exceeds 80A')"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("🧠 AI is analyzing the query..."):
                try:
                    results_df, condition, matched_signal = analyzer.evaluate_condition(query)
                    
                    condition_str = " AND ".join([f"{c['operator']} {c['value']}" for c in condition['conditions']])
                    
                    mapping = {'>': 'greater than', '<': 'less than', '>=': 'greater than or equal to', '<=': 'less than or equal to', '==': 'equal to'}
                    friendly_condition = " and ".join([f"{mapping.get(c['operator'], c['operator'])} {c['value']}" for c in condition['conditions']])
                    duration_str = f"{condition['duration']} seconds" if condition.get('duration', 0) > 0 else "None"
                    msg_id = len(st.session_state.messages)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Target Signal", matched_signal)
                    with col2:
                        st.metric("Condition", condition_str)
                    with col3:
                        st.metric("Duration Constraint", duration_str)
                    
                    if results_df.empty:
                        st.info("No logs matched the specified condition.")
                    else:
                        st.markdown(f"**Answer**: In the following logs the value of **{matched_signal}** is **{friendly_condition}**.")
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name='analysis_results.csv',
                            mime='text/csv',
                            key=f"dl_{msg_id}"
                        )
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "matched_signal": matched_signal,
                        "condition_str": condition_str,
                        "friendly_condition": friendly_condition,
                        "duration_str": duration_str,
                        "results_df": results_df,
                        "id": msg_id
                    })
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "error": f"Error during analysis: {e}"
                    })
