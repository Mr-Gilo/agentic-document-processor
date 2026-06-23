import streamlit as st
import requests
import json
import time

API_URL = "http://127.0.0.1:8080"

st.set_page_config(
    page_title="Agentic Document Processor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic Document Processor")
st.markdown(
    "An AI agent that **plans** which tools to use, **executes** them "
    "in sequence, and **synthesises** a structured report. "
    "Different documents trigger different tool chains."
)

# Sidebar
with st.sidebar:
    st.header("System Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            info = r.json()
            st.success("Agent Online")
            st.markdown(f"**Model:** {info['model']}")
            st.markdown(f"**Pattern:** {info['pattern']}")
            st.markdown(f"**Tools:** {info['tools_available']}")
        else:
            st.error("API Error")
    except Exception:
        st.error("API Offline")
        st.markdown("Start backend:\n```\npython backend/main.py\n```")

    st.divider()
    st.header("Agent Pattern")
    st.markdown("""
Document Input

↓

Phase 1: PLAN

LLM decides which

tools to call

↓

Phase 2: EXECUTE

Tools run in order

Each informs next

↓

Phase 3: SYNTHESISE

LLM compiles report

↓

Structured Output

Execution Trace

""")

    st.divider()
    st.header("Available Tools")
    try:
        r = requests.get(f"{API_URL}/tools", timeout=3)
        if r.status_code == 200:
            for tool in r.json()["tools"]:
                badge = "🔵" if tool["always_run"] else "🟡"
                st.markdown(f"{badge} **{tool['name']}**")
                st.caption(tool["description"])
    except Exception:
        st.markdown("Connect API to see tools")

# Sample documents
SAMPLES = {
    "Select a sample...": "",
    "H&S Incident Report": """INCIDENT REPORT — Health and Safety
Date of Report: 14/03/2024
Location: Packing Hall, North Wing
A serious incident occurred when a forklift operator, T. Williams, 
reported near-miss contact with a pedestrian in a shared access zone.
The site supervisor K. Osei was immediately notified.
The pedestrian, A. Mensah, was uninjured but visibly distressed.
Action taken: Pedestrian barriers installed and forklift traffic
rerouted pending permanent signage installation.
Risk assessment updated and reviewed by safety officer.
RIDDOR report submitted to HSE within 24 hours.""",

    "IT Security Incident": """IT SECURITY INCIDENT REPORT
Reported: 22/05/2024
Affected System: Customer relationship management database
Severity: Critical
Incident: Ransomware alert detected on three endpoints in the sales department.
The Security Operations Centre team identified unusual encryption activity at 02:14.
All affected systems were isolated within 8 minutes of detection.
23,000 customer records may have been exposed prior to containment.
Forensic imaging initiated on all affected workstations.
External cybersecurity firm engaged for full investigation.
CEO and DPO notified. ICO notification being prepared within 72-hour window.""",

    "Field Maintenance Report": """FIELD MAINTENANCE REPORT
Date: 09/01/2024
Site: Site Foxtrot — Bristol
Equipment: Water treatment pump array — Unit 7B
Fault: Catastrophic bearing failure on primary pump detected during 
routine inspection by Senior Engineer M. Abubakar.
Vibration readings 340% above normal operating parameters.
Backup pump brought online at 09:45 to maintain service continuity.
The unit has been offline since 2022 without serviced maintenance.
Replacement bearing assembly ordered from OEM — ETA 72 hours.
Site manager R. Kowalski approved emergency repair budget of £12,500.
Severity: High — production impact minimal due to backup system.""",
}

# Input
col1, col2 = st.columns([2, 1])

with col1:
    selected = st.selectbox("Load a sample document", list(SAMPLES.keys()))
    default_text = SAMPLES[selected]

    document_text = st.text_area(
        "Document text",
        value=default_text,
        height=250,
        placeholder="Paste any document here — incident report, maintenance log, "
                    "security alert, contract, or other text..."
    )

with col2:
    st.markdown("**Document Info**")
    if document_text:
        word_count = len(document_text.split())
        char_count = len(document_text)
        st.metric("Words", word_count)
        st.metric("Characters", char_count)
        st.metric("Estimated tokens", char_count // 4)
    else:
        st.info("Enter a document to see stats")

st.divider()

if st.button("🤖 Process with Agent", type="primary",
             use_container_width=True, disabled=not document_text):
    with st.spinner("Agent is planning, executing, and synthesising..."):
        try:
            start = time.time()
            response = requests.post(
                f"{API_URL}/process",
                json={"text": document_text},
                timeout=180
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()

                st.success(
                    f"Processing complete in {data['total_duration_ms']/1000:.1f}s — "
                    f"{data['tools_called']} tools executed"
                )

                # Final report
                report = data.get("final_report", {})

                st.subheader("📋 Final Report")
                r1, r2, r3 = st.columns(3)
                r1.metric("Document Type",
                          report.get("document_type", "Unknown").replace("_", " ").title())
                r2.metric("Risk Level",
                          report.get("risk_level", "Unknown").upper())
                r3.metric("Tools Called", data["tools_called"])

                st.info(report.get("executive_summary", "No summary available"))

                col_f, col_a = st.columns(2)
                with col_f:
                    st.markdown("**Key Findings**")
                    for finding in report.get("key_findings", []):
                        st.markdown(f"- {finding}")

                with col_a:
                    st.markdown("**Recommended Actions**")
                    for action in report.get("recommended_actions", []):
                        st.markdown(f"- {action}")

                if report.get("anomalies_found"):
                    st.warning(
                        "**Anomalies detected:** " +
                        " | ".join(report["anomalies_found"])
                    )

                st.divider()

                # Agent execution trace
                st.subheader("🔍 Agent Execution Trace")
                st.markdown(
                    "The reasoning steps the agent took — "
                    "showing planning, tool calls, and synthesis."
                )

                trace = data.get("execution_trace", [])
                for step in trace:
                    step_type = step["step_type"]
                    tool = step["tool"]
                    duration = step["duration_ms"]
                    status = step["status"]

                    icon = {"plan": "🧠", "execute": "⚙️",
                            "synthesise": "📝"}.get(step_type, "•")
                    colour = "green" if status == "complete" else "red"

                    label = f"{icon} **{tool.replace('_', ' ').title()}** " \
                            f"— {duration}ms — :{colour}[{status}]"

                    with st.expander(label):
                        st.markdown(f"**Reasoning:** {step['reasoning']}")
                        if step.get("result"):
                            st.json(step["result"])

                st.divider()

                # Plan
                plan = data.get("plan", {})
                with st.expander("🗺️ Agent Plan"):
                    st.markdown(f"**Reasoning:** {plan.get('reasoning', '')}")
                    st.markdown(
                        f"**Tools selected:** "
                        f"`{'` → `'.join(plan.get('selected_tools', []))}`"
                    )
                    st.markdown(
                        f"**Priority:** {plan.get('priority', 'routine').upper()}"
                    )

                # Full JSON
                with st.expander("📦 Full JSON Response"):
                    st.json(data)

            else:
                st.error(
                    f"Error {response.status_code}: "
                    f"{response.json().get('detail', 'Unknown error')}"
                )

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Start it first.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Tab 2: PDF Upload
st.divider()
st.subheader("📄 Upload a PDF Document")
uploaded_pdf = st.file_uploader(
    "Upload a PDF for agentic processing",
    type="pdf",
    key="pdf_upload"
)

if uploaded_pdf:
    if st.button("🤖 Process PDF with Agent", type="secondary",
                 use_container_width=True):
        with st.spinner("Extracting text and processing with agent..."):
            try:
                response = requests.post(
                    f"{API_URL}/process-pdf",
                    files={"file": (uploaded_pdf.name,
                                   uploaded_pdf.getvalue(),
                                   "application/pdf")},
                    timeout=180
                )
                if response.status_code == 200:
                    data = response.json()
                    report = data.get("final_report", {})
                    st.success(
                        f"PDF processed: {data.get('source_filename')} — "
                        f"{data['tools_called']} tools, "
                        f"{data['total_duration_ms']/1000:.1f}s"
                    )
                    st.metric("Risk Level",
                              report.get("risk_level", "Unknown").upper())
                    st.info(report.get("executive_summary", ""))
                    with st.expander("Full Results"):
                        st.json(data)
                else:
                    st.error(response.json().get("detail", "Error"))
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Tab 3: Cache Stats
st.divider()
st.subheader("⚡ Cache Statistics")
try:
    r = requests.get(f"{API_URL}/cache/stats", timeout=3)
    if r.status_code == 200:
        stats = r.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cached Entries", stats["entries"])
        c2.metric("Cache Hits", stats["hits"])
        c3.metric("Cache Misses", stats["misses"])
        c4.metric("Hit Rate", f"{stats['hit_rate']}%")
        if st.button("Clear Cache"):
            requests.delete(f"{API_URL}/cache/clear")
            st.success("Cache cleared")
            st.rerun()
except Exception:
    st.info("Start backend to see cache stats")

# Tab 4: Compare Two Documents
st.divider()
st.subheader("⚖️ Compare Two Documents")
st.markdown("Process two documents and compare their risk profiles side by side.")

col_a, col_b = st.columns(2)
with col_a:
    label_a = st.text_input("Label A", value="Document A")
    doc_a = st.text_area("Document A text", height=150, key="doc_a")
with col_b:
    label_b = st.text_input("Label B", value="Document B")
    doc_b = st.text_area("Document B text", height=150, key="doc_b")

if doc_a and doc_b:
    if st.button("⚖️ Compare Documents", type="secondary",
                 use_container_width=True):
        with st.spinner("Processing both documents..."):
            try:
                response = requests.post(
                    f"{API_URL}/compare",
                    json={
                        "document_a": doc_a,
                        "document_b": doc_b,
                        "label_a": label_a,
                        "label_b": label_b
                    },
                    timeout=300
                )
                if response.status_code == 200:
                    comp = response.json()
                    summary = comp.get("comparison_summary", {})

                    st.success("Comparison complete")
                    m1, m2, m3 = st.columns(3)
                    m1.metric(f"{label_a} Risk",
                              comp["document_a"]["risk_level"].upper()
                              if comp["document_a"]["risk_level"] else "Unknown")
                    m2.metric(f"{label_b} Risk",
                              comp["document_b"]["risk_level"].upper()
                              if comp["document_b"]["risk_level"] else "Unknown")
                    m3.metric("Higher Risk", summary.get("higher_risk", "Unknown"))

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown(f"**{label_a} Key Findings**")
                        for f in comp["document_a"].get("key_findings", []):
                            st.markdown(f"- {f}")
                    with col_r2:
                        st.markdown(f"**{label_b} Key Findings**")
                        for f in comp["document_b"].get("key_findings", []):
                            st.markdown(f"- {f}")

                    with st.expander("Full Comparison JSON"):
                        st.json(comp)
                else:
                    st.error(response.json().get("detail", "Error"))
            except Exception as e:
                st.error(f"Error: {str(e)}")