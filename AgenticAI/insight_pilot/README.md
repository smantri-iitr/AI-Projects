# Insight Pilot

_A lightweight, LLM-powered toolkit that lets analysts explore **any** local dataset with plain-English questions executed through SQL on a DuckDB backend._

## 🚀 What the project does

1. **Exploration Objective Framework**  
   Users can phrase analytical objectives in chat—e.g. “Find drivers of customer churn.” The agent interprets these objectives, determines the relevant tables/fields and plans the analysis end-to-end.

2. **LLM-Guided Insight Discovery**  
   A LangChain + Anthropic powered SQL agent automatically:  
   • inspects the schema  
   • writes and runs valid SQL on DuckDB  
   • returns both the raw results and a concise Markdown summary of the main patterns/trends/anomalies.

3. **Deliverable UI**  
   A minimal Streamlit front-end (`main.py`) exposes the agent through a chat interface so non-technical stakeholders can interact from their browser.

## 🛠️ Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ⚙️ One-time setup

### 1. Obtain an Anthropic API key

Create a free account at anthropic.com and generate an API key. Run the following command on terminal in your project directory:

```bash
echo 'ENTER YOUR API KEY HERE' > ANTHROPIC_API_KEY
```

### 2. Edit `claude.py`

Open `claude.py` and adjust **three placeholders**:

```
db_path = os.path.expanduser('ENTER YOUR DATABASE ABSOLUTE/RELATIVE PATH HERE')
…
llm = ChatAnthropic(model='ENTER YOUR MODEL NAME HERE', temperature=0)
…
query = 'ENTER YOUR QUERY HERE'  # only used when running the file directly
```

## ▶️ Running the app

Start the chat UI with Streamlit:

```bash
streamlit run main.py
```

The browser will open automatically at `http://localhost:8501`.

## ✍️ Project structure

| Path / File        | Purpose                                                             |
| ------------------ | ------------------------------------------------------------------- |
| `claude.py`        | Core function that instantiates the LLM-SQL agent and runs a query. |
| `main.py`          | Streamlit chat interface wiring user prompts → `create_agent()`.    |
| `prompt.txt`       | System prompt that guides the agent’s step-by-step reasoning.       |
| `requirements.txt` | Python dependencies.                                                |
