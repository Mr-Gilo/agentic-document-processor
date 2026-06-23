"""
ReAct Agent Variant for Comparison

ReAct (Reason-Act) loops: the LLM decides the next tool after
each tool execution, rather than planning all tools upfront.

Advantage: More adaptive — can change course based on findings.
Disadvantage: Less reliable with small local models, harder to audit.

Compare with Plan-Execute in agent.py to understand the tradeoffs.
"""

import json
import time
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from tools import (
    classify_document, extract_entities, check_date_consistency,
    flag_anomalies, assess_risk, summarise_document, _extract_json
)
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "llama3.2"
MAX_STEPS = 6


class ReActDocumentAgent:
    """
    ReAct agent: interleaves reasoning and tool calls.
    After each tool result, the LLM decides what to do next.
    """

    TOOL_MAP = {
        "classify_document": classify_document,
        "extract_entities": extract_entities,
        "check_date_consistency": check_date_consistency,
        "flag_anomalies": flag_anomalies,
        "assess_risk": assess_risk,
        "summarise_document": summarise_document,
        "finish": None
    }

    def __init__(self):
        self.llm = ChatOllama(
            model=MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.1,
            num_predict=256
        )
        self.trace: List[Dict] = []
        self.results: Dict[str, Any] = {}

    def _decide_next_tool(self, text: str, history: str) -> Dict:
        """LLM decides next tool based on current results."""
        available = [t for t in self.TOOL_MAP.keys()
                     if t not in self.results or t == "finish"]

        prompt = f"""You are a document analysis agent.
You have run some tools already. Decide what to do next.

Available tools: {available}
Use "finish" when you have enough information.

Previous results summary:
{history[-800:]}

Return ONLY JSON:
{{
  "thought": "What I have learned and what I still need",
  "next_tool": "tool_name or finish",
  "reason": "Why this tool next"
}}"""

        response = self.llm.invoke(prompt)
        decision = _extract_json(response.content)

        if decision is None:
            return {"next_tool": "finish", "thought": "Could not parse",
                    "reason": "Fallback"}
        return decision

    def process(self, text: str) -> Dict[str, Any]:
        """Run ReAct loop: Thought → Action → Observation → repeat."""
        self.trace = []
        self.results = {}
        total_start = time.time()
        history = f"Document excerpt: {text[:300]}\n\nResults so far:\n"

        print(f"\nReAct agent processing ({len(text)} chars)...")

        # Always start with classification
        for tool_name in ["classify_document", "extract_entities"]:
            start = time.time()
            result = self.TOOL_MAP[tool_name](text)
            duration = int((time.time() - start) * 1000)
            self.results[tool_name] = result
            history += f"\n{tool_name}: {json.dumps(result)[:200]}"
            self.trace.append({
                "step": len(self.trace) + 1,
                "tool": tool_name,
                "duration_ms": duration,
                "status": "complete"
            })
            print(f"  [ReAct] {tool_name} ({duration}ms)")

        # ReAct loop for remaining tools
        for step in range(MAX_STEPS):
            decision = self._decide_next_tool(text, history)
            next_tool = decision.get("next_tool", "finish")
            thought = decision.get("thought", "")

            print(f"  [ReAct] Thought: {thought[:80]}")
            print(f"  [ReAct] Next: {next_tool}")

            if next_tool == "finish" or next_tool not in self.TOOL_MAP:
                break

            if next_tool in self.results:
                print(f"  [ReAct] {next_tool} already run, skipping")
                continue

            start = time.time()
            try:
                classification = self.results.get("classify_document")
                entities = self.results.get("extract_entities")
                anomalies = self.results.get("flag_anomalies")

                if next_tool == "check_date_consistency":
                    result = check_date_consistency(text, entities)
                elif next_tool == "flag_anomalies":
                    result = flag_anomalies(text, classification)
                elif next_tool == "assess_risk":
                    result = assess_risk(text, classification, entities, anomalies)
                elif next_tool == "summarise_document":
                    result = summarise_document(text, classification)
                else:
                    result = {"status": "skipped"}

                duration = int((time.time() - start) * 1000)
                self.results[next_tool] = result
                history += f"\n{next_tool}: {json.dumps(result)[:200]}"
                self.trace.append({
                    "step": len(self.trace) + 1,
                    "tool": next_tool,
                    "thought": thought,
                    "duration_ms": duration,
                    "status": "complete"
                })
                print(f"  [ReAct] {next_tool} ({duration}ms)")

            except Exception as e:
                duration = int((time.time() - start) * 1000)
                self.trace.append({
                    "step": len(self.trace) + 1,
                    "tool": next_tool,
                    "duration_ms": duration,
                    "status": "error",
                    "error": str(e)
                })

        total_duration = int((time.time() - total_start) * 1000)

        # Build final report from accumulated results
        risk = self.results.get("assess_risk", {})
        summary = self.results.get("summarise_document", {})
        classification = self.results.get("classify_document", {})

        return {
            "agent_type": "react",
            "success": True,
            "tool_results": self.results,
            "final_report": {
                "document_type": classification.get("document_type", "unknown"),
                "risk_level": risk.get("risk_level", "unknown"),
                "executive_summary": summary.get("summary", ""),
                "key_findings": summary.get("key_findings", []),
                "anomalies_found": self.results.get(
                    "flag_anomalies", {}).get("anomalies", []),
                "recommended_actions": risk.get("recommended_actions", [])
            },
            "execution_trace": self.trace,
            "total_duration_ms": total_duration,
            "tools_called": len(self.results)
        }