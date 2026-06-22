"""
Plan-Execute Document Processing Agent

The agent operates in three phases:
1. PLAN   — LLM reads the document and decides which tools to call
2. EXECUTE — Tools are called in planned order, each output informing the next
3. SYNTHESISE — LLM compiles all results into a final structured report

This is more reliable than ReAct with local models because
tool execution is deterministic — only planning and synthesis use the LLM.
"""

import json
import time
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from tools import (
    classify_document, extract_entities, check_date_consistency,
    flag_anomalies, assess_risk, summarise_document, AVAILABLE_TOOLS, _extract_json
)
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "llama3.2"


class AgentStep:
    """Represents a single step in the agent's execution trace."""

    def __init__(self, step_type: str, tool: str, reasoning: str):
        self.step_type = step_type  # "plan", "execute", "synthesise"
        self.tool = tool
        self.reasoning = reasoning
        self.result = None
        self.duration_ms = 0
        self.status = "pending"

    def to_dict(self):
        return {
            "step_type": self.step_type,
            "tool": self.tool,
            "reasoning": self.reasoning,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "status": self.status
        }


class DocumentProcessingAgent:
    """
    Agentic document processor using the Plan-Execute pattern.

    The agent reasons about documents rather than running a fixed pipeline.
    Different document types trigger different tool chains.
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.1,
            num_predict=512
        )
        self.execution_trace: List[AgentStep] = []

    def _log(self, message: str):
        print(f"  [Agent] {message}")

    def plan(self, text: str) -> Dict[str, Any]:
        """
        Phase 1: LLM decides which tools to call based on document content.
        Returns a plan with tool selection and reasoning.
        """
        self._log("Planning tool sequence...")
        start = time.time()

        tool_descriptions = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in AVAILABLE_TOOLS.items()
        ])

        prompt = f"""You are a document analysis agent.
Read this document excerpt and decide which analysis tools to use.

Available tools:
{tool_descriptions}

Rules for tool selection:
- ALWAYS include: classify_document, extract_entities, summarise_document
- Include check_date_consistency if the document mentions any dates
- Include flag_anomalies if the document mentions incidents, faults, breaches, 
  errors, or anything requiring attention
- Include assess_risk if the document mentions severity, urgency, harm, 
  financial impact, or required actions

Return ONLY JSON:
{{
  "reasoning": "Brief explanation of what this document is and why you chose these tools",
  "selected_tools": ["classify_document", "extract_entities", "flag_anomalies", 
                     "check_date_consistency", "assess_risk", "summarise_document"],
  "expected_findings": "What you expect to find",
  "priority": "routine / elevated / urgent"
}}

Document excerpt (first 400 chars):
{text[:400]}"""

        response = self.llm.invoke(prompt)
        plan = _extract_json(response.content)

        if plan is None:
            plan = {
                "reasoning": "Could not parse planning response — using default tool set",
                "selected_tools": [
                    "classify_document", "extract_entities",
                    "check_date_consistency", "flag_anomalies",
                    "assess_risk", "summarise_document"
                ],
                "expected_findings": "Unknown",
                "priority": "routine"
            }

        # Ensure mandatory tools are always included
        mandatory = ["classify_document", "extract_entities", "summarise_document"]
        for tool in mandatory:
            if tool not in plan["selected_tools"]:
                plan["selected_tools"].insert(0, tool)

        # Filter to valid tools only
        plan["selected_tools"] = [
            t for t in plan["selected_tools"] if t in AVAILABLE_TOOLS
        ]

        duration = int((time.time() - start) * 1000)

        step = AgentStep("plan", "planning", plan["reasoning"])
        step.result = plan
        step.duration_ms = duration
        step.status = "complete"
        self.execution_trace.append(step)

        self._log(f"Plan: {plan['selected_tools']} (priority: {plan.get('priority', 'routine')})")
        return plan

    def execute(self, text: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Execute selected tools in order.
        Each tool's output is passed to subsequent tools as context.
        """
        self._log("Executing tools...")
        results = {}
        classification = None
        entities = None

        for tool_name in plan["selected_tools"]:
            if tool_name not in AVAILABLE_TOOLS:
                continue

            self._log(f"Calling: {tool_name}")
            start = time.time()

            step = AgentStep(
                "execute",
                tool_name,
                f"Executing {tool_name} with context from previous steps"
            )

            try:
                # Pass context from previous tools to enable chaining
                if tool_name == "classify_document":
                    result = classify_document(text)
                    classification = result

                elif tool_name == "extract_entities":
                    result = extract_entities(text)
                    entities = result

                elif tool_name == "check_date_consistency":
                    result = check_date_consistency(text, entities)

                elif tool_name == "flag_anomalies":
                    result = flag_anomalies(text, classification)

                elif tool_name == "assess_risk":
                    anomalies = results.get("flag_anomalies")
                    result = assess_risk(text, classification, entities, anomalies)

                elif tool_name == "summarise_document":
                    result = summarise_document(text, classification)

                else:
                    result = {"status": "skipped", "reason": "unknown tool"}

                duration = int((time.time() - start) * 1000)
                step.result = result
                step.duration_ms = duration
                step.status = "complete"
                results[tool_name] = result
                self._log(f"  {tool_name} completed in {duration}ms")

            except Exception as e:
                duration = int((time.time() - start) * 1000)
                step.result = {"error": str(e)}
                step.duration_ms = duration
                step.status = "error"
                results[tool_name] = {"status": "error", "error": str(e)}
                self._log(f"  {tool_name} failed: {e}")

            self.execution_trace.append(step)

        return results

    def synthesise(self, text: str, results: Dict[str, Any],
                   plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 3: LLM compiles all tool results into a final report.
        """
        self._log("Synthesising final report...")
        start = time.time()

        results_summary = json.dumps(results, indent=2)[:1500]

        prompt = f"""You are a document analysis agent completing your final report.
You have analysed a document using multiple specialised tools.

Tool results summary:
{results_summary}

Based on these results, compile a final report.
Return ONLY valid JSON — every field must be populated:
{{
  "executive_summary": "Write 2-3 sentences summarising what this document 
                        contains and what action is needed",
  "document_type": "the specific document type identified e.g. incident_report",
  "risk_level": "copy from assess_risk results, or infer from flag_anomalies 
                 overall_concern if assess_risk was not run",
  "key_findings": ["Extract 3-5 specific facts found in the document"],
  "entities_identified": {{
    "people": 0,
    "dates": 0,
    "locations": 0
  }},
  "anomalies_found": ["List any anomalies from flag_anomalies results, 
                       or empty list if none"],
  "recommended_actions": ["List 2-3 specific actions from the tool results"],
  "processing_complete": true
}}"""

        response = self.llm.invoke(prompt)
        report = _extract_json(response.content)

        if report is None:
            report = {
                "executive_summary": "Document processed successfully.",
                "document_type": results.get("classify_document", {}).get(
                    "document_type", "unknown"),
                "risk_level": results.get("assess_risk", {}).get(
                    "risk_level", "unknown"),
                "key_findings": [],
                "entities_identified": {},
                "anomalies_found": [],
                "recommended_actions": [],
                "processing_complete": True
            }

        duration = int((time.time() - start) * 1000)

        step = AgentStep("synthesise", "synthesis", "Compiling final report")
        step.result = report
        step.duration_ms = duration
        step.status = "complete"
        self.execution_trace.append(step)

        return report

    def process(self, text: str) -> Dict[str, Any]:
        """
        Full agent pipeline: Plan → Execute → Synthesise.
        Returns complete results including execution trace.
        """
        self.execution_trace = []
        total_start = time.time()

        print(f"\nAgent processing document ({len(text)} chars)...")

        # Phase 1: Plan
        plan = self.plan(text)

        # Phase 2: Execute
        tool_results = self.execute(text, plan)

        # Phase 3: Synthesise
        final_report = self.synthesise(text, tool_results, plan)

        total_duration = int((time.time() - total_start) * 1000)

        return {
            "success": True,
            "plan": plan,
            "tool_results": tool_results,
            "final_report": final_report,
            "execution_trace": [s.to_dict() for s in self.execution_trace],
            "total_duration_ms": total_duration,
            "tools_called": len(tool_results),
            "character_count": len(text)
        }