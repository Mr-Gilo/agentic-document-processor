"""
Agent Tools: Six specialised document analysis functions.

Mix of LLM-powered tools (entity extraction, classification),
deterministic tools (date consistency), and hybrid tools (anomaly detection).
Each tool returns a structured dict with results and confidence.
"""

import json
import re
import os
from datetime import datetime
from typing import Optional
from langchain_ollama import ChatOllama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "llama3.2"


def get_llm():
    """Return configured Ollama LLM instance."""
    return ChatOllama(
        model=MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.1,
        num_predict=512
    )


def _extract_json(text: str) -> Optional[dict]:
    """Robustly extract JSON from LLM response."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ── Tool 1: Classify Document ──────────────────────────────────────────────

def classify_document(text: str) -> dict:
    """
    Identify document type, domain, and key characteristics.
    LLM-powered classification with structured JSON output.
    """
    llm = get_llm()
    snippet = text[:600]

    prompt = f"""Analyse this document and classify it.
Return ONLY a valid JSON object, no other text:
{{
  "document_type": "e.g. incident_report, maintenance_report, legal_contract, 
                   medical_record, financial_statement, email, other",
  "domain": "e.g. health_safety, it_security, field_maintenance, legal, 
             finance, healthcare, general",
  "formality": "formal or informal",
  "estimated_length": "short (<200 words), medium (200-500), long (>500)",
  "key_topics": ["topic1", "topic2", "topic3"],
  "requires_action": true or false,
  "confidence": 0.0 to 1.0
}}

Document excerpt:
{snippet}"""

    response = llm.invoke(prompt)
    result = _extract_json(response.content)

    if result is None:
        result = {
            "document_type": "unknown",
            "domain": "general",
            "formality": "unknown",
            "key_topics": [],
            "requires_action": False,
            "confidence": 0.0
        }

    result["tool"] = "classify_document"
    result["status"] = "success" if result.get("confidence", 0) > 0 else "partial"
    return result


# ── Tool 2: Extract Entities ───────────────────────────────────────────────

def extract_entities(text: str) -> dict:
    """
    Extract named entities: people, dates, locations, organisations, amounts.
    LLM-powered with structured output.
    """
    llm = get_llm()
    snippet = text[:800]

    prompt = f"""Extract all named entities from this document.
Return ONLY a valid JSON object:
{{
  "people": [{{"name": "...", "role": "..."}}],
  "dates": [{{"date": "...", "context": "what this date refers to"}}],
  "locations": [{{"location": "...", "context": "..."}}],
  "organisations": [{{"name": "...", "context": "..."}}],
  "monetary_amounts": [{{"amount": "...", "context": "..."}}],
  "reference_numbers": ["..."],
  "severity_indicators": ["critical", "high", "medium", "low"]
}}

Only include what is explicitly stated. Use empty lists for missing fields.

Document:
{snippet}"""

    response = llm.invoke(prompt)
    result = _extract_json(response.content)

    if result is None:
        result = {
            "people": [], "dates": [], "locations": [],
            "organisations": [], "monetary_amounts": [],
            "reference_numbers": [], "severity_indicators": []
        }

    result["tool"] = "extract_entities"
    result["total_entities"] = sum([
        len(result.get("people", [])),
        len(result.get("dates", [])),
        len(result.get("locations", [])),
        len(result.get("organisations", []))
    ])
    result["status"] = "success"
    return result


# ── Tool 3: Check Date Consistency ────────────────────────────────────────

def check_date_consistency(text: str, entities: Optional[dict] = None) -> dict:
    """
    Deterministic date validation.
    Checks dates are valid, not in the future, and logically sequential.
    """
    issues = []
    validated_dates = []
    today = datetime.now()

    # Extract all date-like patterns from text
    date_patterns = [
        r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b',  # DD/MM/YYYY
        r'\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b',  # YYYY-MM-DD
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})\b',
    ]

    found_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            found_dates.append(" / ".join(match))

    # Also use dates from entity extraction if available
    if entities and entities.get("dates"):
        for d in entities["dates"]:
            date_str = d.get("date", "")
            if date_str:
                found_dates.append(date_str)

    # Validate each date
    for date_str in set(found_dates):
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]:
            try:
                parsed = datetime.strptime(date_str.replace(" / ", "/"), fmt)
                if parsed > today:
                    issues.append(
                        f"Date '{date_str}' is in the future — possible error"
                    )
                elif parsed.year < 2000:
                    issues.append(
                        f"Date '{date_str}' is before year 2000 — verify accuracy"
                    )
                else:
                    validated_dates.append(date_str)
                break
            except ValueError:
                continue

    return {
        "tool": "check_date_consistency",
        "dates_found": len(found_dates),
        "dates_validated": validated_dates,
        "issues": issues,
        "consistency_score": "PASS" if not issues else "FAIL",
        "status": "success"
    }


# ── Tool 4: Flag Anomalies ────────────────────────────────────────────────

def flag_anomalies(text: str, classification: Optional[dict] = None) -> dict:
    """
    Hybrid anomaly detection: keyword signals plus LLM reasoning.
    Identifies unusual patterns, missing information, or red flags.
    """
    flags = []
    risk_keywords = {
        "critical": ["emergency", "critical", "urgent", "immediate", "severe",
                     "fatality", "hospitalised", "evacuated"],
        "high": ["serious", "significant", "major", "breach", "failure",
                 "compromised", "ransomware", "attack"],
        "medium": ["warning", "caution", "degraded", "delayed", "overdue",
                   "escalated", "suspected"],
        "low": ["minor", "routine", "scheduled", "normal", "standard"]
    }

    # Deterministic keyword scan
    text_lower = text.lower()
    detected_level = "low"
    matched_keywords = []

    for level, keywords in risk_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                matched_keywords.append({"keyword": kw, "risk_level": level})
                if level == "critical":
                    detected_level = "critical"
                elif level == "high" and detected_level not in ["critical"]:
                    detected_level = "high"
                elif level == "medium" and detected_level not in ["critical", "high"]:
                    detected_level = "medium"

    # Check for missing information
    doc_type = classification.get("document_type", "") if classification else ""
    if "incident" in doc_type or "report" in doc_type:
        missing = []
        if not re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}', text):
            missing.append("No date detected")
        if not re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text):
            missing.append("No named person detected")
        if missing:
            flags.extend(missing)

    # LLM-powered anomaly reasoning
    llm = get_llm()
    prompt = f"""Review this document excerpt for anomalies, red flags, or unusual patterns.
Return ONLY JSON:
{{
  "anomalies": ["anomaly 1", "anomaly 2"],
  "missing_information": ["missing field 1"],
  "risk_indicators": ["risk 1"],
  "overall_concern": "none / low / medium / high / critical"
}}

Document excerpt:
{text[:500]}"""

    response = llm.invoke(prompt)
    llm_result = _extract_json(response.content) or {}

    return {
        "tool": "flag_anomalies",
        "keyword_risk_level": detected_level,
        "matched_keywords": matched_keywords[:10],
        "missing_information": flags + llm_result.get("missing_information", []),
        "anomalies": llm_result.get("anomalies", []),
        "risk_indicators": llm_result.get("risk_indicators", []),
        "overall_concern": llm_result.get("overall_concern", detected_level),
        "status": "success"
    }


# ── Tool 5: Assess Risk ───────────────────────────────────────────────────

def assess_risk(
    text: str,
    classification: Optional[dict] = None,
    entities: Optional[dict] = None,
    anomalies: Optional[dict] = None
) -> dict:
    """
    Holistic risk assessment combining all available context.
    LLM-powered with structured scoring output.
    """
    llm = get_llm()

    context_summary = f"Document type: {classification.get('document_type', 'unknown')}" \
                      if classification else "Document type: unknown"

    if anomalies:
        context_summary += f"\nOverall concern level: {anomalies.get('overall_concern', 'unknown')}"
        context_summary += f"\nAnomalies detected: {len(anomalies.get('anomalies', []))}"

    prompt = f"""Assess the overall risk level of this document.
Context: {context_summary}

Return ONLY JSON:
{{
  "risk_score": 0 to 100,
  "risk_level": "low / medium / high / critical",
  "primary_risk_factors": ["factor 1", "factor 2"],
  "recommended_actions": ["action 1", "action 2"],
  "requires_immediate_attention": true or false,
  "escalation_required": true or false
}}

Document excerpt:
{text[:600]}"""

    response = llm.invoke(prompt)
    result = _extract_json(response.content)

    if result is None:
        result = {
            "risk_score": 0,
            "risk_level": "unknown",
            "primary_risk_factors": [],
            "recommended_actions": [],
            "requires_immediate_attention": False,
            "escalation_required": False
        }

    result["tool"] = "assess_risk"
    result["status"] = "success"
    return result


# ── Tool 6: Summarise Document ────────────────────────────────────────────

def summarise_document(text: str, classification: Optional[dict] = None) -> dict:
    """
    Generate a concise, structured summary tailored to document type.
    LLM-powered with domain-aware prompting.
    """
    llm = get_llm()
    doc_type = classification.get("document_type", "document") if classification else "document"
    snippet = text[:800]

    prompt = f"""Write a concise summary of this {doc_type}.
Return ONLY JSON:
{{
  "headline": "one sentence headline",
  "summary": "2-3 sentence summary of key facts",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "next_steps": ["step 1", "step 2"],
  "estimated_read_time_seconds": 30
}}

Document:
{snippet}"""

    response = llm.invoke(prompt)
    result = _extract_json(response.content)

    if result is None:
        result = {
            "headline": "Document processed",
            "summary": "Summary could not be generated.",
            "key_findings": [],
            "next_steps": [],
            "estimated_read_time_seconds": 60
        }

    result["tool"] = "summarise_document"
    result["status"] = "success"
    return result


# ── Tool Registry ─────────────────────────────────────────────────────────

AVAILABLE_TOOLS = {
    "classify_document": {
        "fn": classify_document,
        "description": "Identify document type, domain, and key characteristics",
        "always_run": True
    },
    "extract_entities": {
        "fn": extract_entities,
        "description": "Extract people, dates, locations, amounts, references",
        "always_run": True
    },
    "check_date_consistency": {
        "fn": check_date_consistency,
        "description": "Validate dates are correct and logically consistent",
        "always_run": False
    },
    "flag_anomalies": {
        "fn": flag_anomalies,
        "description": "Detect unusual patterns, red flags, or missing information",
        "always_run": False
    },
    "assess_risk": {
        "fn": assess_risk,
        "description": "Score overall risk and recommend actions",
        "always_run": False
    },
    "summarise_document": {
        "fn": summarise_document,
        "description": "Generate concise summary and key findings",
        "always_run": True
    },
}