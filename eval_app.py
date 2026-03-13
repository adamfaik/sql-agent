import streamlit as st
import json
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import os
from agent import create_graph

# 1. Page Configuration
st.set_page_config(page_title="Evaluation Dashboard", page_icon="🧪", layout="wide")
st.title("🧪 LLM Text-to-SQL Evaluation Dashboard")

# --- EXPERIMENT SETTINGS SIDEBAR ---
st.sidebar.header("⚙️ Experiment Settings")
st.sidebar.markdown("Configure the ablation study parameters:")

model_choice = st.sidebar.selectbox(
    "LLM Model",
    options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    help="Select the model used for SQL generation."
)

use_few_shot = st.sidebar.toggle("Enable Few-Shot Prompting", value=True)
use_self_correction = st.sidebar.toggle("Enable Self-Correction Loop", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Disable components one by one to measure their individual impact on the final accuracy.")
# ----------------------------------------

st.markdown("Run automated benchmarks to measure the accuracy, robustness, and self-correction rate of the LangGraph agent.")

# 2. Load the Benchmark Dataset
BENCHMARK_PATH = "eval/benchmark.json"

@st.cache_data
def load_benchmark():
    """Loads the benchmark JSON file."""
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

benchmark_data = load_benchmark()

if not benchmark_data:
    st.error(f"Benchmark file not found at {BENCHMARK_PATH}. Please create it.")
    st.stop()

# Display the dataset in an expander for transparency
with st.expander("📂 View Benchmark Dataset", expanded=False):
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)

# 3. Evaluation Logic
if st.button("🚀 Run Evaluation Benchmark", type="primary"):
    
    agent_graph = create_graph()
    results_log = []
    
    # Initialize metric counters
    total_questions = len(benchmark_data)
    valid_sql_count = 0
    accurate_count = 0
    self_corrected_count = 0
    
    # UI Elements for progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    for i, item in enumerate(benchmark_data):
        status_text.write(f"Evaluating Question {item['id']}/{total_questions}: *{item['question']}*")
        
        initial_state = {"query": item['question']}

        # Isolate context and PASS HYPERPARAMETERS dynamically
        config = {
            "configurable": {
                "thread_id": f"eval_thread_{item['id']}",
                "model_name": model_choice,
                "use_few_shot": use_few_shot,
                "use_self_correction": use_self_correction
            }
        }
        
        # Invoke the LangGraph agent with the config
        final_state = agent_graph.invoke(initial_state, config=config)  
        
        # Extract metadata from the final state
        generated_sql = final_state.get("sql_query", "")
        error = final_state.get("error", "")
        retries = final_state.get("retry_count", 0)
        complexity = final_state.get("query_complexity", "unknown")
        
        # Metric 1: Valid SQL (Did it produce a query without crashing SQLite?)
        is_valid_sql = error == "" and generated_sql != ""
        
        # Metric 2: Execution Accuracy (Does the output match the Gold SQL?)
        is_accurate = False
        
        if item["difficulty"] == "out_of_scope":
            # Success for out_of_scope is correctly identifying it and NOT generating SQL
            is_accurate = (complexity == "out_of_scope")
            
        elif is_valid_sql:
            valid_sql_count += 1
            if retries > 0:
                self_corrected_count += 1
                
            try:
                # Compare database outputs
                conn = sqlite3.connect("olist.db")
                cursor = conn.cursor()
                
                cursor.execute(item["gold_sql"])
                gold_results = cursor.fetchall()
                
                cursor.execute(generated_sql)
                agent_results = cursor.fetchall()
                conn.close()
                
                # Strict evaluation: Row results must match exactly
                if gold_results == agent_results:
                    is_accurate = True
            except Exception:
                pass # Execution failed during evaluation
                
        # INCROYABLEMENT IMPORTANT : On incrémente le compteur global à la fin !
        if is_accurate:
            accurate_count += 1
                
        # Log the result for the final dataframe
        results_log.append({
            "ID": item["id"],
            "Difficulty": item["difficulty"],
            "Accurate": "✅ Yes" if is_accurate else "❌ No",
            "Valid SQL": "✅ Yes" if is_valid_sql or item["difficulty"] == "out_of_scope" else "❌ No",
            "Retries Triggered": retries,
            "Final Error": error if error else "None"
        })
        
        # Update progress
        progress_bar.progress((i + 1) / total_questions)

    status_text.success("✅ Evaluation Complete!")
    
    # 4. Display Metrics Dashboard
    st.markdown("---")
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate percentages
    accuracy_rate = (accurate_count / total_questions) * 100
    valid_sql_rate = ((valid_sql_count + sum(1 for d in benchmark_data if d['difficulty'] == 'out_of_scope')) / total_questions) * 100
    
    with col1:
        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = accuracy_rate,
            title = {'text': "Execution Accuracy (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00cc96"}}
        ))
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = valid_sql_rate,
            title = {'text': "Valid SQL Rate (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#636efa"}}
        ))
        st.plotly_chart(fig2, use_container_width=True)
        
    with col3:
        st.metric(label="🔄 Self-Correction Triggers", value=self_corrected_count, delta="Saved from failure", delta_color="normal")
        st.markdown("*Number of times the agent successfully rewrote broken SQL using the error loop.*")

    # 5. Detailed Results Table
    st.subheader("📝 Detailed Test Logs")
    df_results = pd.DataFrame(results_log)
    st.dataframe(df_results, use_container_width=True)
    
    # Optional Export
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results (CSV)", csv, "evaluation_results.csv", "text/csv")