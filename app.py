import streamlit as st
from agent import create_graph

# 1. Page Configuration
st.set_page_config(page_title="Olist Data Agent", page_icon="🤖", layout="wide")
st.title("🤖 Olist Text-to-SQL Agent")
st.markdown("Ask me anything about the Olist e-commerce database! I will translate your question into SQL, execute it, and summarize the results.")

# 2. Initialize the LangGraph Agent in Streamlit's Session State
if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = create_graph()

# 3. Initialize UI Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display previous conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Chat Input Area
if prompt := st.chat_input("Ex: How many customers are in Sao Paulo?"):
    
    # Display the user's question immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing Area
    with st.chat_message("assistant"):
        # Visual indicator of the pipeline steps
        status = st.status("🧠 Agent is thinking...", expanded=True)
        
        initial_state = {"query": prompt}
        final_summary = ""
        
        # 6. Stream the graph execution to capture intermediate results
        # .stream() yields the output of each node as it finishes
        for event in st.session_state.agent_graph.stream(initial_state):
            for node_name, node_state in event.items():
                
                # --- NODE 1: SQL GENERATION ---
                if node_name == "generate_sql":
                    status.write("⏳ Translating to SQL...")
                    with st.expander("🔍 View Generated SQL", expanded=False):
                        # Syntax highlighting for SQL
                        st.code(node_state.get("sql_query"), language="sql")
                        
                # --- NODE 2: DATABASE EXECUTION ---
                elif node_name == "execute_sql":
                    error = node_state.get("error")
                    if error:
                        status.error(f"⚠️ SQL Error detected: {error}. Routing back to correct it...")
                    else:
                        status.write("⏳ Executing query on database...")
                        with st.expander("📊 View Raw Database Results", expanded=False):
                            st.text(node_state.get("db_results"))
                            
                # --- NODE 3: SUMMARIZATION ---
                elif node_name == "summarize_results":
                    status.write("⏳ Synthesizing final answer...")
                    final_summary = node_state.get("summary")

        # Update the status box when finished
        status.update(label="✅ Task Completed!", state="complete", expanded=False)
        
        # 7. Display the final natural language response
        st.markdown(final_summary)
        st.session_state.messages.append({"role": "assistant", "content": final_summary})