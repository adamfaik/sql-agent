import streamlit as st
import pandas as pd
import uuid
from agent import create_graph
from langchain_core.messages import HumanMessage
import plotly.express as px

st.set_page_config(page_title="Olist Data Agent", page_icon="🤖", layout="wide")
st.title("🤖 Olist Text-to-SQL Agent")
st.markdown("Ask me anything about the Olist e-commerce database! I can write SQL, execute it, self-correct errors, and analyze the results.")

# 1. Initialize session states
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4()) # Unique session ID for memory

if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = create_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there's an attached dataframe in history, display it
        if "dataframe" in msg:
            st.dataframe(msg["dataframe"], use_container_width=True)

# 3. Chat Input
if prompt := st.chat_input("Ex: What are the top 5 product categories by revenue?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("🧠 Agent is thinking...", expanded=True)
        
        initial_state = {
            "query": prompt, 
            "messages": [HumanMessage(content=prompt)] # Inject the question into the agent's memory
            }
        
        # Configuration required for the Checkpointer to track this specific conversation
        config = {"configurable": {"thread_id": st.session_state.thread_id}} 
        
        final_summary = ""
        final_df = None
        
        # Stream the execution
        for event in st.session_state.agent_graph.stream(initial_state, config=config):
            for node_name, node_state in event.items():
                
                if node_name == "generate_sql":
                    status.write("⏳ Translating to SQL...")
                    with st.expander("🔍 View Generated SQL", expanded=False):
                        st.code(node_state.get("sql_query"), language="sql")
                        
                elif node_name == "execute_sql":
                    error = node_state.get("error")
                    if error:
                        status.error(f"⚠️ SQL Error: {error}. Routing back to LLM to self-correct...")
                    else:
                        status.write("⏳ Executing query on database...")
                        raw_data = node_state.get("raw_data", [])
                        if raw_data:
                            final_df = pd.DataFrame(raw_data)
                            with st.expander("📊 View Raw Data", expanded=False):
                                st.dataframe(final_df, use_container_width=True)
                            
                elif node_name == "summarize_results":
                    status.write("⏳ Synthesizing final answer...")
                    final_summary = node_state.get("summary")

        status.update(label="✅ Task Completed!", state="complete", expanded=False)
        
        # Display the interactive dataframe if data was retrieved
        if final_df is not None and not final_df.empty:
            st.dataframe(final_df, use_container_width=True)
            
            # --- NEW: AUTOMATIC DATA VISUALIZATION ---
            # We only try to plot if there's more than 1 row and at least 2 columns
            if len(final_df) > 1 and len(final_df.columns) >= 2:
                # Auto-detect column types
                numeric_cols = final_df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = final_df.select_dtypes(exclude=['number']).columns.tolist()
                
                # If we have both categories and numbers, draw a chart!
                if numeric_cols and categorical_cols:
                    x_col = categorical_cols[0] # Use the first text column for the X axis
                    y_col = numeric_cols[0]     # Use the first number column for the Y axis
                    
                    st.markdown(f"📊 **Visualisation : {y_col} par {x_col}**")
                    
                    # Create a beautiful interactive Bar Chart using Plotly
                    fig = px.bar(
                        final_df, 
                        x=x_col, 
                        y=y_col, 
                        color=x_col,
                        text_auto='.2s', # Show values on top of bars
                        template="plotly_dark" # Matches Streamlit's dark mode nicely
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
        st.markdown(final_summary)