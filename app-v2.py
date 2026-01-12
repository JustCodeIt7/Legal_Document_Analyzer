import os
import tempfile
import hashlib
from textwrap import dedent
from typing import TypedDict, Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langchain_docling.loader import DoclingLoader, ExportType

################################ Configuration & Setup ################################

load_dotenv()

DEFAULT_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_MAX_INPUT_CHARS = 10_000
DEFAULT_CHUNK_CHARS = 3_500
DEFAULT_CHUNK_OVERLAP = 250


class AgentState(TypedDict, total=False):
    original_text: str
    condensed_text: str
    summary: str
    risks: str
    suggestions: str
    final_report: str


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= overlap:
        overlap = 0
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


@st.cache_resource
def get_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def call_ollama(prompt: str, model: str, base_url: str, temperature: float) -> str:
    llm = get_llm(model, base_url, temperature)
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()


def map_reduce_condense(
    text: str,
    model: str,
    base_url: str,
    temperature: float,
    chunk_chars: int,
    overlap: int,
) -> str:
    chunks = chunk_text(text, chunk_chars, overlap)

    partials = []
    for i, c in enumerate(chunks, start=1):
        prompt = dedent(f"""
        You are an expert legal assistant.
        Summarize this portion of a legal document. Preserve obligations, deadlines, termination, payment, liability, indemnity, IP, confidentiality, governing law, dispute resolution.
        Output as concise bullet points.

        [Chunk {i}/{len(chunks)}]
        {c}
        """)
        partials.append(call_ollama(prompt, model, base_url, temperature))

    reduce_prompt = dedent(f"""
    You are an expert legal assistant.
    Consolidate the following chunk summaries into a single coherent, non-redundant summary.
    Use short sections with headings:
    - Parties & Purpose
    - Key Obligations
    - Money (fees, payment terms, penalties)
    - Term, Renewal, Termination
    - Liability, Indemnity, Insurance
    - IP & Confidentiality
    - Disputes & Governing Law
    - Important Deadlines

    Chunk summaries:
    {"\n\n".join(partials)}
    """)
    return call_ollama(reduce_prompt, model, base_url, temperature)


################################ Graph Node Functions ################################

def summarize_node(state: AgentState) -> Dict[str, Any]:
    condensed = state.get("condensed_text") or state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Produce a concise executive summary of this legal document (5–12 bullet points max).
    Focus on practical meaning and major commitments.

    Document:
    {condensed}
    """)
    return {
        "summary": call_ollama(
            prompt, st.session_state.model, OLLAMA_BASE_URL, st.session_state.temp_summary
        )
    }


def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    condensed = state.get("condensed_text") or state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Identify key legal risks and liabilities. For each item include:
    - Risk
    - Why it matters
    - Severity: Low/Med/High
    - Likelihood: Low/Med/High
    - Suggested mitigation (1 line)

    Document:
    {condensed}
    """)
    return {
        "risks": call_ollama(
            prompt, st.session_state.model, OLLAMA_BASE_URL, st.session_state.temp_risks
        )
    }


def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    condensed = state.get("condensed_text") or state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Suggest improvements or missing protections. Prefer specific clause-level suggestions.
    Organize by topic (payment, termination, limitation of liability, indemnity, confidentiality, IP, dispute resolution, warranties).

    Document:
    {condensed}
    """)
    return {
        "suggestions": call_ollama(
            prompt,
            st.session_state.model,
            OLLAMA_BASE_URL,
            st.session_state.temp_suggestions,
        )
    }


def compile_report_node(state: AgentState) -> Dict[str, Any]:
    report = dedent(f"""
    # Legal Document Analysis (AI-Assisted)

    > Disclaimer: This is not legal advice. Consult a qualified attorney before acting.

    ## 📝 Document Summary
    {state.get("summary","")}

    ## ⚠️ Identified Risks
    {state.get("risks","")}

    ## 💡 Suggestions for Improvement
    {state.get("suggestions","")}
    """).strip()
    return {"final_report": report}


@st.cache_resource
def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("suggest_improvements", suggest_improvements_node)
    workflow.add_node("compile_report", compile_report_node)

    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "analyze_risks")
    workflow.add_edge("analyze_risks", "suggest_improvements")
    workflow.add_edge("suggest_improvements", "compile_report")
    workflow.add_edge("compile_report", END)
    return workflow.compile()


################################ Helper Functions ################################

def load_doc(uploaded_file) -> str:
    suffix = uploaded_file.name.split(".")[-1].lower()
    data = uploaded_file.getvalue()

    if suffix == "txt":
        return data.decode("utf-8", errors="replace")

    tmp_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    loader = DoclingLoader(file_path=tmp_path, export_type=ExportType.MARKDOWN)
    docs = loader.load()
    text = "\n\n".join(d.page_content or "" for d in docs).strip()

    os.remove(tmp_path)
    return text


def analyze_document(doc_text: str) -> Dict[str, Any]:
    app = get_workflow()
    return app.invoke({"original_text": doc_text, "condensed_text": st.session_state.condensed_text})


################################ Main App Interface ################################

def main():
    st.set_page_config(page_title="Legal Doc Analyzer", layout="wide")

    st.markdown(
        """
        <style>
        .report-box {
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            background: rgba(255,255,255,0.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload Legal Document (PDF or TXT)", type=["pdf", "txt"])

        st.divider()
        st.subheader("Model Settings")
        st.session_state.model = st.text_input("Model", value=DEFAULT_MODEL)
        st.session_state.temp_summary = st.slider("Summary temperature", 0.0, 1.0, 0.2, 0.05)
        st.session_state.temp_risks = st.slider("Risks temperature", 0.0, 1.0, 0.2, 0.05)
        st.session_state.temp_suggestions = st.slider("Suggestions temperature", 0.0, 1.0, 0.3, 0.05)

        st.divider()
        st.subheader("Long Document Handling")
        max_chars = st.number_input("Max extracted chars (preview)", value=DEFAULT_MAX_INPUT_CHARS, step=1000)
        chunk_chars = st.number_input("Chunk size (chars)", value=DEFAULT_CHUNK_CHARS, step=500)
        overlap = st.number_input("Chunk overlap (chars)", value=DEFAULT_CHUNK_OVERLAP, step=50)

    st.title("⚖️ AI Legal Document Analyzer")

    if uploaded_file:
        data = uploaded_file.getvalue()
        h = file_hash(data)
        st.session_state.setdefault("results_by_hash", {})

        if st.button("Analyze Document"):
            prog = st.progress(0, text="Reading document...")
            doc_text = load_doc(uploaded_file)

            preview_text = doc_text[: int(max_chars)]
            with st.expander("Extracted text preview"):
                st.text(preview_text)

            prog.progress(25, text="Condensing long document...")
            condensed = map_reduce_condense(
                doc_text,
                model=st.session_state.model,
                base_url=OLLAMA_BASE_URL,
                temperature=0.2,
                chunk_chars=int(chunk_chars),
                overlap=int(overlap),
            )
            st.session_state.condensed_text = condensed

            prog.progress(55, text="Running analysis agents...")
            result = analyze_document(doc_text)

            prog.progress(100, text="Done.")
            st.session_state.results_by_hash[h] = result

        if h in st.session_state.results_by_hash:
            result = st.session_state.results_by_hash[h]

            st.markdown("## 📊 Analysis Report")
            summary_tab, risks_tab, suggestions_tab, report_tab = st.tabs(
                ["Summary", "Risks", "Suggestions", "Full report"]
            )

            with summary_tab:
                st.markdown("### 📝 Document Summary")
                st.markdown(f"<div class='report-box'>{result.get('summary','')}</div>", unsafe_allow_html=True)

            with risks_tab:
                st.markdown("### ⚠️ Identified Risks")
                st.markdown(f"<div class='report-box'>{result.get('risks','')}</div>", unsafe_allow_html=True)

            with suggestions_tab:
                st.markdown("### 💡 Suggestions for Improvement")
                st.markdown(f"<div class='report-box'>{result.get('suggestions','')}</div>", unsafe_allow_html=True)

            with report_tab:
                st.markdown(result.get("final_report", ""))

            st.download_button("Download Report", result.get("final_report", ""), file_name="legal_analysis.md")

    else:
        st.warning("Upload a PDF or TXT document to begin.")


if __name__ == "__main__":
    main()