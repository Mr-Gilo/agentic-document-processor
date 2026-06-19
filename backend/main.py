from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import DocumentProcessingAgent
import uvicorn
import os

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


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=False)