# Agentic Document Processor

An AI agent that dynamically plans and executes multi-step document analysis
using the Plan-Execute pattern. Different documents trigger different tool chains.

Unlike fixed pipelines, the agent reasons about what to do — reading the document
first, deciding which tools are relevant, executing them in order with context
passing between steps, then synthesising a structured report.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

## Agent Pattern

Document Input

↓

Phase 1: PLAN

LLM reads document excerpt

Selects relevant tools

Sets processing priority

↓

Phase 2: EXECUTE

Tools run in planned order

Each tool's output passed

as context to the next

↓

Phase 3: SYNTHESISE

LLM compiles all results

into structured report

↓

Final Report + Execution Trace

## Why Plan-Execute over ReAct?

ReAct (Reason-Act) loops require the LLM to decide the next action after
each tool call. With smaller local models (llama3.2, 3B parameters), this
leads to unreliable tool selection mid-loop. Plan-Execute separates reasoning
from execution: the LLM plans once upfront, then tools execute deterministically.
This gives reliable, auditable behaviour on consumer hardware.

## Six Specialised Tools

| Tool | Type | Description |
|------|------|-------------|
| classify_document | LLM | Identify type, domain, formality, key topics |
| extract_entities | LLM | People, dates, locations, amounts, references |
| check_date_consistency | Deterministic | Validate dates are correct and sequential |
| flag_anomalies | Hybrid (keyword + LLM) | Detect red flags and missing information |
| assess_risk | LLM | Score severity and recommend actions |
| summarise_document | LLM | Concise summary with key findings |

**Mandatory tools** (always run): classify_document, extract_entities, summarise_document

**Conditional tools** (agent decides): check_date_consistency, flag_anomalies, assess_risk

## Sample Documents

Three built-in samples demonstrate different tool chains:

- **H&S Incident Report** — triggers all 6 tools, CRITICAL risk level
- **IT Security Incident** — triggers anomaly detection, escalation flags
- **Field Maintenance Report** — triggers date validation, equipment risk scoring

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangChain + Plan-Execute pattern |
| LLM | Ollama (llama3.2) — fully local |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Document Loading | PyMuPDF |

## Prerequisites

- Python 3.11+
- Ollama installed: https://ollama.com/download
- llama3.2 model: `ollama pull llama3.2`

## Installation

```bash
git clone https://github.com/Mr-Gilo/agentic-document-processor.git
cd agentic-document-processor

conda create -n agent-processor python=3.11 -y
conda activate agent-processor
pip install -r requirements.txt
```

## Running

**Terminal 1 — Backend:**
```bash
conda activate agent-processor
cd backend
python main.py
```
Backend: http://127.0.0.1:8080
API docs: http://127.0.0.1:8080/docs

**Terminal 2 — Frontend:**
```bash
conda activate agent-processor
streamlit run app.py
```
Frontend: http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Agent status and tool count |
| POST | /process | Submit document for agentic processing |
| GET | /tools | List all available tools |

## Example Output

```json
{
  "plan": {
    "selected_tools": ["classify_document", "extract_entities",
                       "flag_anomalies", "check_date_consistency",
                       "assess_risk", "summarise_document"],
    "priority": "urgent"
  },
  "final_report": {
    "document_type": "incident_report",
    "risk_level": "critical",
    "executive_summary": "Safety incident involving forklift near-miss...",
    "key_findings": ["Forklift operator T. Williams involved in near-miss"],
    "anomalies_found": ["Forklift operation in shared access zone"],
    "recommended_actions": ["Conduct thorough risk assessment"]
  },
  "execution_trace": [
    {"tool": "planning", "duration_ms": 11082, "status": "complete"},
    {"tool": "classify_document", "duration_ms": 3136, "status": "complete"},
    {"tool": "extract_entities", "duration_ms": 3877, "status": "complete"},
    {"tool": "flag_anomalies", "duration_ms": 3322, "status": "complete"},
    {"tool": "check_date_consistency", "duration_ms": 0, "status": "complete"},
    {"tool": "assess_risk", "duration_ms": 3549, "status": "complete"},
    {"tool": "summarise_document", "duration_ms": 3677, "status": "complete"},
    {"tool": "synthesis", "duration_ms": 5395, "status": "complete"}
  ]
}
```

## Applying to New Domains

The agent is domain-agnostic. To adapt for motor insurance court pack analysis:
replace or extend the sample documents with court pack examples. The six tools
apply directly: classify (claim type), extract entities (parties, dates, amounts),
check date consistency (incident vs filing dates), flag anomalies (inconsistencies),
assess risk (fraud indicators), summarise (claim overview).

## Related Projects

- [pdf-extractor](https://github.com/Mr-Gilo/pdf-extractor) — Fixed pipeline PDF extraction
- [rag-document-assistant](https://github.com/Mr-Gilo/rag-document-assistant) — RAG document Q&A
- [multimodal-risk-pipeline](https://github.com/Mr-Gilo/multimodal-risk-pipeline) — Risk scoring
- [document-extraction-finetuning](https://github.com/Mr-Gilo/document-extraction-finetuning) — LoRA fine-tuning

## Roadmap

- [x] Plan-Execute agent pattern
- [x] Six specialised tools (LLM, deterministic, hybrid)
- [x] Context passing between tools
- [x] Execution trace with timing
- [x] FastAPI backend with Swagger docs
- [x] Streamlit frontend with reasoning visualisation
- [ ] PDF upload support
- [ ] Tool result caching
- [ ] Multi-document comparison
- [ ] ReAct agent variant for comparison