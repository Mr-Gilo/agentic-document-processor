from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import DocumentProcessingAgent
import uvicorn
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from document_loader import extract_text_from_pdf

app = FastAPI(
    title="Agentic Document Processor API",
    description=(
        "AI agent that plans and executes multi-step document analysis. "
        "Uses the Plan-Execute pattern with six specialised tools."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "llama3.2",
        "pattern": "plan-execute",
        "tools_available": 6
    }


class DocumentRequest(BaseModel):
    text: str
    document_hint: str = ""


@app.post("/process")
async def process_document(request: DocumentRequest):
    """
    Submit a document for agentic processing.
    The agent will plan which tools to use, execute them,
    and return a synthesised report with full execution trace.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")

    if len(request.text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Document too long. Maximum 10,000 characters."
        )

    try:
        agent = DocumentProcessingAgent()
        result = agent.process(request.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}"
        )

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and process it with the agentic pipeline.
    Text is extracted then passed to the full Plan-Execute agent.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text extracted. PDF may be image-based or scanned."
            )

        if len(text) > 10000:
            text = text[:10000]

        agent = DocumentProcessingAgent()
        result = agent.process(text)
        result["source_filename"] = file.filename
        result["extracted_characters"] = len(text)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )

@app.post("/process-react")
async def process_with_react(request: DocumentRequest):
    """
    Process document using ReAct agent pattern.
    Compare with /process (Plan-Execute) to see the difference.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")

    try:
        from react_agent import ReActDocumentAgent
        agent = ReActDocumentAgent()
        result = agent.process(request.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ReAct processing failed: {str(e)}"
        )

@app.get("/tools")
def list_tools():
    """List all available agent tools and descriptions."""
    from tools import AVAILABLE_TOOLS
    return {
        "tools": [
            {
                "name": name,
                "description": info["description"],
                "always_run": info["always_run"]
            }
            for name, info in AVAILABLE_TOOLS.items()
        ]
    }

@app.get("/cache/stats")
def cache_stats():
    """Return cache hit rate and entry count."""
    from cache import tool_cache
    return tool_cache.stats()

@app.delete("/cache/clear")
def clear_cache():
    """Clear the tool result cache."""
    from cache import tool_cache
    tool_cache.clear()
    return {"status": "cleared"}

class CompareRequest(BaseModel):
    document_a: str
    document_b: str
    label_a: str = "Document A"
    label_b: str = "Document B"


@app.post("/compare")
async def compare_documents(request: CompareRequest):
    """
    Process two documents and return a structured comparison.
    Useful for comparing similar incident reports or claim documents.
    """
    if not request.document_a.strip() or not request.document_b.strip():
        raise HTTPException(
            status_code=400,
            detail="Both documents must contain text"
        )

    try:
        agent_a = DocumentProcessingAgent()
        agent_b = DocumentProcessingAgent()

        result_a = agent_a.process(request.document_a)
        result_b = agent_b.process(request.document_b)

        report_a = result_a.get("final_report", {})
        report_b = result_b.get("final_report", {})

        comparison = {
            "label_a": request.label_a,
            "label_b": request.label_b,
            "document_a": {
                "document_type": report_a.get("document_type"),
                "risk_level": report_a.get("risk_level"),
                "executive_summary": report_a.get("executive_summary"),
                "key_findings": report_a.get("key_findings", []),
                "anomalies_found": report_a.get("anomalies_found", []),
                "recommended_actions": report_a.get("recommended_actions", []),
                "tools_called": result_a.get("tools_called"),
                "duration_ms": result_a.get("total_duration_ms")
            },
            "document_b": {
                "document_type": report_b.get("document_type"),
                "risk_level": report_b.get("risk_level"),
                "executive_summary": report_b.get("executive_summary"),
                "key_findings": report_b.get("key_findings", []),
                "anomalies_found": report_b.get("anomalies_found", []),
                "recommended_actions": report_b.get("recommended_actions", []),
                "tools_called": result_b.get("tools_called"),
                "duration_ms": result_b.get("total_duration_ms")
            },
            "comparison_summary": {
                "same_document_type": (
                    report_a.get("document_type") == report_b.get("document_type")
                ),
                "risk_levels": {
                    request.label_a: report_a.get("risk_level"),
                    request.label_b: report_b.get("risk_level")
                },
                "higher_risk": (
                    request.label_a
                    if _risk_rank(report_a.get("risk_level")) >=
                       _risk_rank(report_b.get("risk_level"))
                    else request.label_b
                ),
                "total_anomalies": (
                    len(report_a.get("anomalies_found", [])) +
                    len(report_b.get("anomalies_found", []))
                )
            }
        }

        return comparison

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


def _risk_rank(level: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(
        str(level).lower(), 0
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=False)