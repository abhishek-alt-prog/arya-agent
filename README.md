# Arya Learning System - AI Agent

The AI Agent for the Arya Learning System. This service is responsible for dynamically generating curriculum content, evaluating progress, and adapting the difficulty of upcoming lessons using a local Large Language Model (Gemma 4 via Ollama).

## Tech Stack
* **Language:** Python 3.11+
* **LLM Engine:** Ollama (running locally)
* **Model:** Google Gemma 4 (27B parameter)
* **Libraries:** Pydantic (data modeling), Requests (HTTP client), Streamlit (Parent Dashboard)

## Prerequisites
* Python 3.11+
* [Ollama](https://ollama.com/) running locally with the `gemma3:27b` model pulled.
* The **BFF Service** must be running and accessible.

## Setup

1. **Install dependencies:**
   We recommend using a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   *(Or if using pip directly from the `pyproject.toml` definition: `pip install requests pydantic pydantic-settings streamlit`)*

2. **Configuration:**
   Copy `.env.example` to `.env` and update the values:
   ```env
   BFF_BASE_URL=http://localhost:8080
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=gemma3:27b
   ```

## Running the Agent

The agent is designed to be run as a CLI tool (e.g., via a cron job or manual trigger).

### 1. Initial Setup
Generates the initial course skeletons and the first batch of EASY lessons for a new student.
```bash
python -m src.main --child-id <CHILD_ID> setup
```

### 2. Nightly Adaptation
Analyses recent quiz results, determines mastery scores, and generates upcoming lessons. 
- High mastery -> Advances to next topic / Increases difficulty.
- Low mastery -> Simplifies current topic.
```bash
python -m src.main --child-id <CHILD_ID> adapt
```

### 3. Health Check
Check connectivity to the BFF and Ollama.
```bash
python -m src.main --child-id <CHILD_ID> status
```

## Parent Dashboard
The agent repository also includes a Streamlit dashboard designed for parents to monitor their child's progress, view detailed mastery scores, and manually trigger the agent.

```bash
streamlit run src/dashboard.py
```
