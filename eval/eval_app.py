import streamlit as st
import json
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from agent import create_graph
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import uuid
import time
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Pydantic Model for the Judge ---
class JudgeOutput(BaseModel):
    score: int = Field(description="Score from 0 to 3")
    reason: str = Field(description="Short explanation for the score given")

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
use_cot_planner = st.sidebar.toggle("Enable CoT SQL Planner", value=True)

st.sidebar.markdown("---")

# --- Cache clearing button for active development ---
if st.sidebar.button("🔄 Reload Benchmark Dataset"):
    st.cache_data.clear() # Clears the Streamlit cache
    st.rerun()            # Refreshes the UI immediately

st.sidebar.info("Tip: Disable components one by one to measure their individual impact on the final accuracy.")

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

    # Start timer for the whole evaluation run
    run_start_time = time.time()
    
    for i, item in enumerate(benchmark_data):
        status_text.write(f"Evaluating Question {item['id']}/{total_questions}: *{item['question']}*")
        
        initial_state = {"query": item['question']}

        # Isolate context and PASS HYPERPARAMETERS dynamically
        config = {
            "configurable": {
                "thread_id": f"eval_thread_{item['id']}",
                "model_name": model_choice,
                "use_few_shot": use_few_shot,
                "use_self_correction": use_self_correction,
                "use_cot_planner": use_cot_planner
            }
        }
        
        # Invoke the LangGraph agent with the config
        final_state = agent_graph.invoke(initial_state, config=config)  
        
        # Extract metadata from the final state
        generated_sql = final_state.get("sql_query", "")
        error = final_state.get("error", "")
        retries = final_state.get("retry_count", 0)
        complexity = final_state.get("query_complexity", "unknown")
        
        # Initialize result variables to prevent NameError if execution fails
        gold_results = []
        agent_results = []
        
        # Metric 1: Valid SQL (Did it produce a query without crashing SQLite?)
        is_valid_sql = error == "" and generated_sql != ""
        
        # Metric 2: Execution Accuracy (Does the output match the Gold SQL?)
        is_accurate = False
        
        if item.get("difficulty") == "out_of_scope":
            # Success for out_of_scope is correctly identifying it and NOT generating SQL
            is_accurate = (complexity == "out_of_scope")
            
        elif item.get("difficulty") == "ambiguous":
            # --- Ambiguous queries have no exact "Gold" answer ---
            # We execute the agent's SQL to see if it works, but we DO NOT compare it
            is_accurate = False # Strict match is fundamentally N/A for ambiguity
            if is_valid_sql:
                valid_sql_count += 1
                if retries > 0:
                    self_corrected_count += 1
                # Safe connection handling (prevents memory leaks)
                conn = None
                try:
                    conn = sqlite3.connect("data/olist.db")
                    cursor = conn.cursor()
                    cursor.execute(generated_sql)
                    agent_results = cursor.fetchall()
                except Exception:
                    pass
                finally:
                    # Guarantee connection closure even if fetchall() fails
                    if conn:
                        conn.close()
            
        elif is_valid_sql:
            valid_sql_count += 1
            if retries > 0:
                self_corrected_count += 1
                
            try:
                # Use context manager to guarantee the connection is closed
                # even if an exception is raised during query execution
                with sqlite3.connect("olist.db") as conn:
                    cursor = conn.cursor()
        
                    # Fetch the reference results from the hand-written gold SQL
                    if item.get("gold_sql"):
                        cursor.execute(item["gold_sql"])
                        gold_results = cursor.fetchall()
        
                    # Fetch the agent's results using the LLM-generated SQL
                    cursor.execute(generated_sql)
                    agent_results = cursor.fetchall()
    
                # Strict exact-match evaluation: both result sets must be identical
                if gold_results == agent_results:
                    is_accurate = True
            except Exception:
                pass
                
        # Metric 3: Self-Correction Rate is tracked via retries and self_corrected_count
        if is_accurate:
            accurate_count += 1
        
        # LLM-as-a-Judge (Semantic Eval)
        
        # Guard clause for out_of_scope queries to prevent arbitrary judge scoring
        if item.get("difficulty") == "out_of_scope":
            # Direct scoring: 3 points if correctly identified, 0 otherwise
            judge_score = 3 if is_accurate else 0
            judge_reason = "Out-of-scope detection: correct classification." if is_accurate else "Failed to identify out-of-scope query."
            
        elif item.get("difficulty") == "ambiguous":
            # --- FIX: Special Judge Prompt for Ambiguity ---
            judge_llm = ChatOpenAI(model="gpt-4o", temperature=0) 
            structured_judge = judge_llm.with_structured_output(JudgeOutput)
            
            agent_str = str(agent_results[:5]) if agent_results else "No Agent results"
            
            judge_prompt = f"""You are an impartial SQL evaluation judge.
            The user asked an AMBIGUOUS question: "{item.get('question', 'N/A')}"
            There is no single correct 'Gold SQL' because the question is open to interpretation (e.g., 'best' could mean highest revenue, largest volume, or best ratings).
            
            Agent generated SQL: {generated_sql}
            Agent SQL result (first 5 rows): {agent_str}
            
            Score the agent on a scale of 0 to 3 based on the LOGIC and REASONABLENESS of its assumption:
            - 3: Highly reasonable business interpretation and valid SQL returning data.
            - 2: Plausible interpretation but minor SQL flaw or weird sorting.
            - 1: Poor interpretation but valid SQL.
            - 0: Completely illogical interpretation or invalid/no SQL generated.
            """
            
            try:
                judge_res = structured_judge.invoke([HumanMessage(content=judge_prompt)])
                judge_score = judge_res.score
                judge_reason = judge_res.reason
            except Exception as e:
                judge_score = 0
                judge_reason = f"Judge Evaluation Error: {str(e)}"

        else:
            # Standard evaluation for actual SQL queries
            # We use a structured output to guarantee valid JSON formatting
            judge_llm = ChatOpenAI(model="gpt-4o", temperature=0) 
            structured_judge = judge_llm.with_structured_output(JudgeOutput)
            
            # Format results for the prompt (limit to 5 rows to save tokens/context)
            gold_str = str(gold_results[:5]) if gold_results else "No Gold SQL provided or executed"
            agent_str = str(agent_results[:5]) if agent_results else "No Agent results"
            
            judge_prompt = f"""You are an impartial SQL evaluation judge.
            Your task: determine if the agent's SQL answer is semantically correct for the user's question.
            
            User question: {item.get('question', 'N/A')}
            Gold SQL result: {gold_str}
            Agent SQL result: {agent_str}
            
            Score the agent on a scale of 0 to 3:
            - 3: Perfectly correct (same data, possibly different order/format)
            - 2: Partially correct (right direction, missing a filter or wrong aggregation)
            - 1: Wrong but related (queried the right tables but wrong logic)
            - 0: Completely wrong or no SQL generated
            """
            
            try:
                # Invoke the judge
                judge_res = structured_judge.invoke([HumanMessage(content=judge_prompt)])
                judge_score = judge_res.score
                judge_reason = judge_res.reason
            except Exception as e:
                # Fallback if the judge fails
                judge_score = 0
                judge_reason = f"Judge Evaluation Error: {str(e)}"
                
        # Log the result for the final dataframe
        results_log.append({
            "ID": item["id"],
            "Difficulty": item["difficulty"],
            "Accurate (Strict)": "✅ Yes" if is_accurate else "❌ No",
            "Judge Score (/3)": judge_score,
            "Judge Reason": judge_reason,
            "Valid SQL": "✅ Yes" if is_valid_sql or item["difficulty"] == "out_of_scope" else "❌ No",
            "Retries Triggered": retries,
            "Final Error": error if error else "None"
        })
        
        # Update progress
        progress_bar.progress((i + 1) / total_questions)
        
        # Rate Limit Prevention: Add a short delay after each question, especially for more powerful models
        if model_choice == "gpt-4o":
            time.sleep(12)

    status_text.success("✅ Evaluation Complete!")
    
    # 4. Display Metrics Dashboard
    st.markdown("---")
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate percentages
    accuracy_rate = (accurate_count / total_questions) * 100 if total_questions > 0 else 0
    
    # Calculate Valid SQL Rate properly
    # Only count out_of_scope cases that were correctly rejected (no SQL generated)
    correctly_rejected = sum(
        1 for log in results_log 
        if log.get("Difficulty") == "out_of_scope" and log.get("Accurate (Strict)") == "✅ Yes"
    )
    valid_sql_rate = ((valid_sql_count + correctly_rejected) / total_questions) * 100 if total_questions > 0 else 0
    
    # Calculate Semantic Rate
    total_judge_score = sum(row["Judge Score (/3)"] for row in results_log)
    max_possible_score = total_questions * 3
    semantic_accuracy_rate = (total_judge_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    with col1:
        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = accuracy_rate,
            title = {'text': "Strict Accuracy (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00cc96"}}
        ))
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # New Gauge for Semantic Accuracy
        fig_semantic = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = semantic_accuracy_rate,
            title = {'text': "Semantic Accuracy (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#ff9900"}}
        ))
        st.plotly_chart(fig_semantic, use_container_width=True)
        
    with col3:
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = valid_sql_rate,
            title = {'text': "Valid SQL Rate (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#636efa"}}
        ))
        st.plotly_chart(fig2, use_container_width=True)
        
    with col4:
        st.metric(label="🔄 Self-Correction Triggers", value=self_corrected_count, delta="Saved from failure", delta_color="normal")
        st.markdown("*Number of times the agent successfully rewrote broken SQL using the error loop.*")

    # 5. Detailed Results Table
    st.subheader("📝 Detailed Test Logs")
    df_results = pd.DataFrame(results_log)
    st.dataframe(df_results, use_container_width=True)
    
    # Optional Export
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results (CSV)", csv, "evaluation_results.csv", "text/csv")

    # Save Run History (Ablation Study)    
    # Calculate average execution time per question
    run_end_time = time.time()
    total_time = run_end_time - run_start_time
    avg_time = total_time / total_questions if total_questions > 0 else 0
    
    RUNS_HISTORY_PATH = "eval/runs_history.json"
    
    # Create the metadata dictionary for this specific run
    run_metadata = {
        "Run ID": str(uuid.uuid4())[:8],
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_choice,
        "Few-Shot": use_few_shot,
        "Self-Correction": use_self_correction,
        "CoT Planner": use_cot_planner,
        "Strict Accuracy (%)": round(accuracy_rate, 1),
        "Semantic Accuracy (%)": round(semantic_accuracy_rate, 1),
        "Valid SQL Rate (%)": round(valid_sql_rate, 1),
        "Self-Corrections": self_corrected_count,
        "Avg Time/Q (s)": round(avg_time, 2)
    }
    
    # Load existing history if the file exists
    if os.path.exists(RUNS_HISTORY_PATH):
        with open(RUNS_HISTORY_PATH, "r", encoding="utf-8") as f:
            try:
                runs_history = json.load(f)
            except json.JSONDecodeError:
                runs_history = []
    else:
        runs_history = []
        
    # Append the current run metrics and save back to disk
    runs_history.append(run_metadata)
    
    with open(RUNS_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(runs_history, f, indent=4)

    # UI: Display Ablation Study Comparison
    st.markdown("---")
    st.subheader("📈 Ablation Study — Configurations Comparison")
    st.markdown("Compare the performance of different model configurations and features across historical runs. Perfect for academic research and reporting.")
    
    # Convert history to DataFrame and display
    df_history = pd.DataFrame(runs_history)
    
    # Display the dataframe with the most recent runs at the top
    st.dataframe(df_history.iloc[::-1], use_container_width=True)
    
    # Optional export for the historical data
    csv_history = df_history.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Run History (CSV)", csv_history, "ablation_study_history.csv", "text/csv")