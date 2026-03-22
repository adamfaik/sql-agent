# 🤖 Text-to-SQL multi-agent architecture evaluation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)

## 📌 Project overview
This repository contains the source code and experimental framework for evaluating multi-agent architectures in text-to-SQL tasks. It was developed as part of a final thesis project.

The core objective is to evaluate how advanced prompt engineering techniques—specifically **few-shot learning**, **chain of thought (CoT)**, and **self-correction**—impact the performance and reliability of an LLM (powered by `gpt-4o-mini` and `LangGraph`) when querying a complex, real-world relational database.

## 🗄️ The dataset
The agent is tested against the [Brazilian e-commerce public dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This realistic dataset contains 100,000 anonymized orders from 2016 to 2018.
- The database schema consists of 9 interconnected tables (orders, customers, products, reviews, etc.).
- An in-depth **Exploratory Data Analysis (EDA)** and Entity-Relationship Diagram (ERD) can be found in the `notebooks/` directory to justify the architectural choices.

## 🏗️ Project structure
* `agent.py`: The core logic defining the LangGraph state machine, nodes, and LLM integrations.
* `app.py`: A Streamlit interactive chat interface acting as the glass box to demonstrate the agent's reasoning, SQL generation, and auto-correction.
* `scripts/`: Utility scripts to build the SQLite database (`build_db.py`) and extract schema context (`build_schema.py`).
* `eval/`: Contains the evaluation framework.
  * `benchmark.json`: A ground-truth benchmark of 60 questions categorized by difficulty (simple, complex, ambiguous, out-of-scope).
  * `eval_app.py`: A Streamlit dashboard to run automated evaluations and track metrics (accuracy, latency, token usage).
* `notebooks/`: Jupyter notebooks detailing the data exploration (`eda_olist.ipynb`) and thesis statistical analysis.

## 🚀 Setup and installation

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/sql-agent.git](https://github.com/adamfaik/sql-agent.git)
cd sql-agent
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Environment variables**
Create a `.env` file in the root directory and add your OpenAI API Key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

**5. Initialize the database**
Before running the agent, you must build the SQLite database from the raw CSV files (assuming you have downloaded the Olist dataset into a `data/` folder).
```bash
python scripts/build_db.py
python scripts/build_schema.py
```

## 🎯 Usage

**Run the interactive agent app:**
```bash
streamlit run app.py
```

**Run the evaluation dashboard:**
```bash
streamlit run eval/eval_app.py
```

## 🔬 Key findings
Through rigorous ablation studies, this project demonstrates that while self-correction significantly reduces syntax errors (SQLite exceptions), complex relational joins still require robust schema linking and few-shot examples to prevent semantic hallucinations.

---
*Developed for academic training purposes.*