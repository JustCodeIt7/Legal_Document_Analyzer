import os
import tempfile
import hashlib
from textwrap import dedent
from typing import TypedDict, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import pymupdf4llm

################################ Configuration & Setup ################################

# Load environment variables from .env file
load_dotenv()

DEFAULT_MODEL = "granite4:1b"
# Retrieve OLLAMA_BASE_URL from environment, default to localhost
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

################################ Data Structures ################################


class AgentState(TypedDict, total=False):
    original_text: str
    summary: str
    risks: str
    suggestions: str
    final_report: str


################################ Core Utilities ################################


def file_hash(data: bytes) -> str:
    # Generate a SHA256 hash for given binary data
    return hashlib.sha256(data).hexdigest()


@st.cache_resource
def get_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    # Initialize and cache the Ollama LLM instance
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def call_ollama(prompt: str, model: str, base_url: str, temperature: float) -> str:
    # Invoke the LLM with a given prompt and retrieve its content
    llm = get_llm(model, base_url, temperature)
    resp = llm.invoke(prompt)
    # Extract and strip the response content
    return (resp.content or "").strip()


################################ AI Agent Nodes ################################


def summarize_node(state: AgentState) -> Dict[str, Any]:
    # Extract the original document text for processing
    text = state["original_text"]
    # Craft a prompt for the LLM to generate a concise executive summary
    prompt = dedent(f"""
    You are an expert legal assistant.
    Produce a concise executive summary of this legal document (5–12 bullet points max).
    Focus on practical meaning and major commitments.

    Document:
    {text}
    """)
    # Call the LLM to get the summary, using a specific temperature setting
    return {
        "summary": call_ollama(
            prompt,
            st.session_state.model,
            OLLAMA_BASE_URL,
            st.session_state.temp_summary,
        )
    }


def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    # Extract the original document text for risk analysis
    text = state["original_text"]
    # Formulate a prompt to identify key legal risks and liabilities
    prompt = dedent(f"""
    You are an expert legal assistant.
    Identify key legal risks and liabilities. For each item include:
    - Risk
    - Why it matters
    - Severity: Low/Med/High
    - Likelihood: Low/Med/High
    - Suggested mitigation (1 line)

    Document:
    {text}
    """)
    # Call the LLM to get the risk analysis, using a specific temperature setting
    return {
        "risks": call_ollama(
            prompt, st.session_state.model, OLLAMA_BASE_URL, st.session_state.temp_risks
        )
    }


def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    # Extract the original document text for improvement suggestions
    text = state["original_text"]
    # Create a prompt to suggest improvements or missing protections
    prompt = dedent(f"""
    You are an expert legal assistant.
    Suggest improvements or missing protections. Prefer specific clause-level suggestions.
    Organize by topic (payment, termination, limitation of liability, indemnity, confidentiality, IP, dispute resolution, warranties).

    Document:
    {text}
    """)
    # Call the LLM to get suggestions, using a specific temperature setting
    return {
        "suggestions": call_ollama(
            prompt,
            st.session_state.model,
            OLLAMA_BASE_URL,
            st.session_state.temp_suggestions,
        )
    }


def compile_report_node(state: AgentState) -> Dict[str, Any]:
    # Assemble the final report from the generated summary, risks, and suggestions
    report = dedent(f"""
    # Legal Document Analysis (AI-Assisted)

    > Disclaimer: This is not legal advice. Consult a qualified attorney before acting.

    ## 📝 Document Summary
    {state.get("summary", "")}

    ## ⚠️ Identified Risks
    {state.get("risks", "")}

    ## 💡 Suggestions for Improvement
    {state.get("suggestions", "")}
    """).strip()
    # Store the complete report in the state
    return {"final_report": report}


################################ Workflow Management ################################


@st.cache_resource
def get_workflow():
    # Initialize a StateGraph for orchestrating agent tasks
    workflow = StateGraph(AgentState)
    # Define individual nodes for each analytical step
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("suggest_improvements", suggest_improvements_node)
    workflow.add_node("compile_report", compile_report_node)

    # Set the starting point of the workflow
    workflow.set_entry_point("summarize")
    # Define the sequence of execution for the nodes
    workflow.add_edge("summarize", "analyze_risks")
    workflow.add_edge("analyze_risks", "suggest_improvements")
    workflow.add_edge("suggest_improvements", "compile_report")
    # Mark the end of the workflow
    workflow.add_edge("compile_report", END)

    # Compile the graph into an executable application
    return workflow.compile()


################################ Document Handling ################################


def load_doc(uploaded_file) -> str:
    # Determine file type based on extension
    suffix = uploaded_file.name.split(".")[-1].lower()
    # Read the content of the uploaded file
    data = uploaded_file.getvalue()

    # Handle plain text files directly
    if suffix == "txt":
        return data.decode("utf-8", errors="replace")

    # Process PDF files by saving temporarily and converting to markdown
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    # Convert the temporary file to markdown text using pymupdf4llm
    text = pymupdf4llm.to_markdown(tmp_path)
    # Clean up the temporary file
    os.remove(tmp_path)
    # Return the extracted text, stripped of leading/trailing whitespace
    return text.strip()


def analyze_document(doc_text: str) -> Dict[str, Any]:
    # Get the pre-configured workflow application
    app = get_workflow()
    # Invoke the workflow with the document text as initial state
    return app.invoke({"original_text": doc_text})


################################ Streamlit Application ################################


def display_analysis_results(h):
    result = st.session_state.results_by_hash[h]

    st.markdown("## 📊 Analysis Report")
    # Create tabs to organize the different parts of the report
    summary_tab, risks_tab, suggestions_tab, report_tab = st.tabs(
        ["Summary", "Risks", "Suggestions", "Full report"]
    )

    # Display the document summary
    with summary_tab:
        st.markdown("### 📝 Document Summary")
        st.markdown(result.get("summary", ""))

    # Display the identified risks
    with risks_tab:
        st.markdown("### ⚠️ Identified Risks")
        st.markdown(result.get("risks", ""))

    # Display the suggestions for improvement
    with suggestions_tab:
        st.markdown("### ### 💡 Suggestions for Improvement")
        st.markdown(result.get("suggestions", ""))
    # Display the complete compiled report
    with report_tab:
        st.markdown(result.get("final_report", ""))

    # Provide a button to download the full report
    st.download_button(
        "Download Report", result.get("final_report", ""), file_name="legal_analysis.md"
    )


def analyze_uploaded_file(uploaded_file, h):
    # Show progress during document processing
    prog = st.progress(0, text="Reading document...")
    # Load and extract text from the uploaded document
    doc_text = load_doc(uploaded_file)

    preview_lines = doc_text.splitlines()
    preview_text = "\n".join(preview_lines[:30])
    with st.expander("Document preview (first 30 lines)"):
        st.markdown(preview_text if preview_text else "[No content extracted]")

    prog.progress(50, text="Running analysis agents...")
    # Run the AI workflow to analyze the document
    result = analyze_document(doc_text)

    prog.progress(100, text="Done.")
    # Cache the analysis result using the file hash
    st.session_state.results_by_hash[h] = result


def render_sidebar_inputs():
    st.header("Upload Document")
    # Allow users to upload PDF or TXT files
    result = st.file_uploader("Upload Legal Document (PDF or TXT)", type=["pdf", "txt"])

    st.divider()
    st.subheader("Model Settings")
    # Allow user to specify the LLM model name
    st.session_state.model = st.text_input("Model", value=DEFAULT_MODEL)
    # Configure temperature for summary generation
    st.session_state.temp_summary = st.slider(
        "Summary temperature", 0.0, 1.0, 0.2, 0.05
    )
    # Configure temperature for risk analysis
    st.session_state.temp_risks = st.slider("Risks temperature", 0.0, 1.0, 0.2, 0.05)
    # Configure temperature for suggestions
    st.session_state.temp_suggestions = st.slider(
        "Suggestions temperature", 0.0, 1.0, 0.3, 0.05
    )

    return result


def main():
    # Configure the Streamlit page title and layout
    st.set_page_config(page_title="Legal Doc Analyzer", layout="wide")

    # Create a sidebar for file upload and model settings
    with st.sidebar:
        uploaded_file = render_sidebar_inputs()
    st.title("⚖️ AI Legal Document Analyzer")

    # Display UI only if a file has been uploaded
    if uploaded_file:
        # Get raw data and compute its hash for caching results
        data = uploaded_file.getvalue()
        h = file_hash(data)
        # Initialize session state for storing analysis results by hash
        st.session_state.setdefault("results_by_hash", {})

        # Trigger document analysis when the button is clicked
        if st.button("Analyze Document"):
            analyze_uploaded_file(uploaded_file, h)
        # Display analysis results if available for the current file
        if h in st.session_state.results_by_hash:
            display_analysis_results(h)
    else:
        st.warning("Upload a PDF or TXT document to begin.")


# Ensure the main function runs when the script is executed
if __name__ == "__main__":
    main()
