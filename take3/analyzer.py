import os
os.environ["HF_HOME"] = "./local_models"
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import pipeline
import json
import re
import numpy as np

class LogAnalyzer:
    def __init__(self, data_paths, embedding_model="all-MiniLM-L6-v2", llm_model="Qwen/Qwen2.5-0.5B-Instruct"):
        self.data_paths = data_paths
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load LLM pipeline (on CPU)
        print(f"Loading {llm_model}...")
        self.llm = pipeline("text-generation", model=llm_model, device="cpu", torch_dtype=torch.float32)
        print("Model loaded.")

        # Read header and identify signals
        print("Analyzing CSV structure...")
        df_header = pd.read_csv(self.data_paths[0], encoding='latin1', nrows=0)
        self.columns = list(df_header.columns)
        
        # Extract signal columns and map them to their corresponding time columns
        self.signals = []
        self.signal_to_col_index = {}
        for i, col in enumerate(self.columns):
            if "Unnamed:" not in col and not col.startswith("Time[s]"):
                self.signals.append(col)
                self.signal_to_col_index[col] = i

        # Embed all signal names for RAG
        print(f"Embedding {len(self.signals)} signal names for search...")
        self.signal_embeddings = self.embedding_model.encode(self.signals)
        print("Initialization complete.")

    def retrieve_signal(self, query_signal_name):
        """Finds the closest matching signal in the CSV based on semantic similarity."""
        query_embedding = self.embedding_model.encode([query_signal_name])
        similarities = cos_sim(query_embedding, self.signal_embeddings)[0]
        best_idx = torch.argmax(similarities).item()
        return self.signals[best_idx]

    def parse_query_with_llm(self, query):
        """Uses the local LLM to parse the query into a structured JSON condition."""
        system_prompt = (
            "You are a helpful data analysis assistant. Extract parameters from the user's query into a strict JSON format.\n"
            "The JSON must contain exactly these keys:\n"
            "- \"signal\": The name of the signal being queried (string).\n"
            "- \"conditions\": A list of condition objects, where each object has:\n"
            "   - \"operator\": The mathematical operator (must be one of '>', '<', '>=', '<=', '==').\n"
            "   - \"value\": The numerical threshold value (float).\n"
            "- \"duration\": The duration in seconds the condition must hold (float). If not specified, use 0.0.\n\n"
            "Respond ONLY with the JSON object. Do not add markdown or explanations."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        result = self.llm(messages, max_new_tokens=100, temperature=0.1)
        raw_output = result[0]['generated_text'][-1]['content']
        
        # Clean the output in case the model added markdown blocks
        json_str = raw_output.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed = json.loads(json_str)
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {raw_output}. Error: {e}")

    def evaluate_condition(self, query):
        """Parses the query, retrieves the signal, reads data, and finds matching logs."""
        print("Parsing query...")
        condition = self.parse_query_with_llm(query)
        print(f"Parsed condition: {condition}")
        
        # RAG Step: Match signal name
        matched_signal = self.retrieve_signal(condition['signal'])
        print(f"Matched signal: {matched_signal}")
        
        # Load the data for this specific signal
        sig_idx = self.signal_to_col_index[matched_signal]
        time_idx = sig_idx - 1 # Time column is typically immediately preceding the signal column
        
        time_col = self.columns[time_idx]
        val_col = matched_signal
        

        duration = condition.get('duration', 0.0)
        
        all_results = []
        
        for file_path in self.data_paths:
            log_name = os.path.basename(file_path)
            print(f"Reading data for '{matched_signal}' from {log_name}...")
            try:
                # Only load the relevant columns to save memory
                df = pd.read_csv(file_path, encoding='latin1', usecols=[time_col, val_col])
                
                # Clean data (drop NaNs due to different cycle times)
                df = df.dropna(subset=[val_col]).sort_values(by=time_col)
                
                times = df[time_col].values
                values = df[val_col].values
                
                # Apply boolean condition
                bool_arr = np.ones(len(values), dtype=bool)
                for cond in condition['conditions']:
                    op = cond['operator']
                    thresh = cond['value']
                    if op == '>':
                        bool_arr &= (values > thresh)
                    elif op == '<':
                        bool_arr &= (values < thresh)
                    elif op == '>=':
                        bool_arr &= (values >= thresh)
                    elif op == '<=':
                        bool_arr &= (values <= thresh)
                    elif op == '==':
                        bool_arr &= (values == thresh)
                    else:
                        raise ValueError(f"Unknown operator {op}")
                    
                if duration <= 0:
                    # Simple threshold check
                    matched_indices = np.where(bool_arr)[0]
                    for idx in matched_indices:
                        all_results.append({"Log_Name": log_name, "Start_Time": times[idx], "End_Time": times[idx], "Signal": matched_signal, "Value": values[idx]})
                else:
                    # Duration check (greater than X for more than Y seconds)
                    in_block = False
                    block_start_t = None
                    
                    for i, is_true in enumerate(bool_arr):
                        t = times[i]
                        if is_true and not in_block:
                            in_block = True
                            block_start_t = t
                        elif not is_true and in_block:
                            in_block = False
                            block_end_t = times[i-1]
                            if (block_end_t - block_start_t) > duration:
                                all_results.append({"Log_Name": log_name, "Start_Time": block_start_t, "End_Time": block_end_t, "Signal": matched_signal, "Duration": block_end_t - block_start_t})
                    
                    # Check last block
                    if in_block:
                        block_end_t = times[-1]
                        if (block_end_t - block_start_t) > duration:
                            all_results.append({"Log_Name": log_name, "Start_Time": block_start_t, "End_Time": block_end_t, "Signal": matched_signal, "Duration": block_end_t - block_start_t})
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        return pd.DataFrame(all_results), condition, matched_signal
