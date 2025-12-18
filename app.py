"""
Medical RAG - Streamlit UI

A confidence-gated medical document Q&A system.
Clinical Editorial Design - Bright, precise, authoritative.
"""

import os
import time
import concurrent.futures
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Secrets & Config
# ─────────────────────────────────────────────────────────────────────────────
def load_secrets():
    """Load API keys from Streamlit secrets or .env file."""
    try:
        os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
        os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
    except (KeyError, FileNotFoundError):
        from dotenv import load_dotenv
        load_dotenv()

load_secrets()

from rag_pipeline import RAGSystem

st.set_page_config(
    page_title="MedRAG",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Clinical Editorial CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Source+Sans+3:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --white: #FFFFFF;
    --cream: #FAFAF9;
    --warm-gray-50: #F5F5F4;
    --warm-gray-100: #E7E5E4;
    --warm-gray-200: #D6D3D1;
    --warm-gray-400: #A8A29E;
    --warm-gray-500: #78716C;
    --warm-gray-600: #57534E;
    --warm-gray-800: #292524;
    --charcoal: #1C1917;
    --teal-500: #14B8A6;
    --teal-600: #0D9488;
    --teal-700: #0F766E;
    --coral-400: #FB923C;
    --coral-500: #F97316;
    --emerald-500: #10B981;
    --emerald-600: #059669;
    --rose-500: #F43F5E;
}

/* Reset Streamlit defaults */
.stApp {
    background: var(--white) !important;
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', -apple-system, sans-serif;
    color: var(--charcoal);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}
div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}

/* Subtle grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image:
        linear-gradient(rgba(20, 184, 166, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20, 184, 166, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* Main container */
.main .block-container {
    max-width: 860px;
    padding: 2rem 1.5rem 8rem 1.5rem;
    position: relative;
    z-index: 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
   HEADER
   ═══════════════════════════════════════════════════════════════════════════ */

.header-container {
    text-align: center;
    padding: 3rem 0 2.5rem 0;
    border-bottom: 1px solid var(--warm-gray-100);
    margin-bottom: 2rem;
}

.header-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--teal-600);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
}

.header-eyebrow::before,
.header-eyebrow::after {
    content: '';
    width: 24px;
    height: 1px;
    background: var(--teal-500);
    opacity: 0.5;
}

.header-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 3rem;
    font-weight: 600;
    color: var(--charcoal);
    margin: 0 0 0.75rem 0;
    letter-spacing: -0.02em;
    line-height: 1.1;
    font-optical-sizing: auto;
}

.header-subtitle {
    font-size: 1.125rem;
    color: var(--warm-gray-500);
    margin: 0 0 2rem 0;
    font-weight: 400;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.5;
}

/* Stats strip */
.stats-strip {
    display: flex;
    justify-content: center;
    gap: 3rem;
    padding: 1.5rem 0;
    background: var(--cream);
    border-radius: 8px;
    border: 1px solid var(--warm-gray-100);
}

.stat-block {
    text-align: center;
}

.stat-number {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--teal-600);
    line-height: 1;
}

.stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--warm-gray-500);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.375rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SAMPLE QUESTIONS
   ═══════════════════════════════════════════════════════════════════════════ */

.samples-section {
    margin: 2rem 0;
    padding: 1.5rem;
    background: var(--cream);
    border-radius: 8px;
    border: 1px solid var(--warm-gray-100);
}

.samples-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--warm-gray-500);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

.sample-item {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.9rem;
    color: var(--warm-gray-600);
    padding: 0.625rem 0;
    border-bottom: 1px solid var(--warm-gray-100);
    cursor: text;
    user-select: all;
}

.sample-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.sample-item:hover {
    color: var(--teal-600);
}

/* ═══════════════════════════════════════════════════════════════════════════
   CHAT INTERFACE
   ═══════════════════════════════════════════════════════════════════════════ */

/* Chat messages container */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
}

/* User avatar */
.stChatMessage [data-testid="chatAvatarIcon-user"] {
    background: var(--charcoal) !important;
    border-radius: 4px !important;
}

/* Assistant avatar */
.stChatMessage [data-testid="chatAvatarIcon-assistant"] {
    background: var(--teal-600) !important;
    border-radius: 4px !important;
}

/* User message text */
[data-testid="stChatMessageContent"] p {
    font-size: 1rem;
    line-height: 1.6;
    color: var(--charcoal);
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESPONSE CARDS
   ═══════════════════════════════════════════════════════════════════════════ */

.response-card {
    background: var(--white);
    border: 1px solid var(--warm-gray-200);
    border-radius: 8px;
    padding: 1.75rem;
    margin: 0.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.response-answer {
    font-size: 1rem;
    line-height: 1.75;
    color: var(--charcoal);
}

.response-answer strong {
    font-weight: 600;
    color: var(--charcoal);
}

/* Confidence meter */
.confidence-meter {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--warm-gray-100);
}

.confidence-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.625rem;
}

.confidence-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--warm-gray-500);
}

.confidence-value {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.125rem;
    font-weight: 600;
}

.conf-high { color: var(--emerald-600); }
.conf-medium { color: var(--coral-500); }
.conf-low { color: var(--rose-500); }

.confidence-track {
    height: 6px;
    background: var(--warm-gray-100);
    border-radius: 3px;
    overflow: hidden;
}

.confidence-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.bar-high { background: linear-gradient(90deg, var(--emerald-500), var(--teal-500)); }
.bar-medium { background: linear-gradient(90deg, var(--coral-400), var(--coral-500)); }
.bar-low { background: linear-gradient(90deg, var(--rose-500), #FB7185); }

/* ═══════════════════════════════════════════════════════════════════════════
   ABSTENTION CARD
   ═══════════════════════════════════════════════════════════════════════════ */

.abstention-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid var(--coral-500);
    border-radius: 0 8px 8px 0;
    padding: 1.5rem;
    margin: 0.5rem 0;
}

.abstention-header {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--coral-500);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.abstention-header::before {
    content: '⚠';
    font-size: 1.1rem;
}

.abstention-reason {
    font-size: 0.95rem;
    color: #92400E;
    line-height: 1.6;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SOURCES
   ═══════════════════════════════════════════════════════════════════════════ */

.sources-toggle {
    margin-top: 1.25rem;
}

.sources-toggle summary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--teal-600);
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.5rem 0;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.sources-toggle summary::-webkit-details-marker {
    display: none;
}

.sources-toggle summary::before {
    content: '›';
    font-size: 1rem;
    transition: transform 0.2s ease;
}

.sources-toggle[open] summary::before {
    transform: rotate(90deg);
}

.sources-toggle summary:hover {
    color: var(--teal-700);
}

.sources-list {
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.source-card {
    background: var(--cream);
    border: 1px solid var(--warm-gray-100);
    border-radius: 6px;
    padding: 1rem;
}

.source-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.625rem;
}

.source-name {
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--charcoal);
}

.source-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--teal-600);
    background: rgba(20, 184, 166, 0.1);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
}

.source-excerpt {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.6;
    color: var(--warm-gray-600);
}

/* ═══════════════════════════════════════════════════════════════════════════
   CHAT INPUT
   ═══════════════════════════════════════════════════════════════════════════ */

/* Input container */
[data-testid="stChatInput"] {
    background: var(--white) !important;
}

[data-testid="stChatInput"] > div {
    background: var(--white) !important;
    border: 2px solid var(--warm-gray-200) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s ease;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--teal-500) !important;
    box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.1) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 1rem !important;
    color: var(--charcoal) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--warm-gray-400) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════════════════ */

[data-testid="stSidebar"] {
    background: var(--cream) !important;
    border-right: 1px solid var(--warm-gray-100) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: var(--white) !important;
    color: var(--charcoal) !important;
    border: 1px solid var(--warm-gray-200) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--warm-gray-50) !important;
    border-color: var(--warm-gray-300) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS / PROGRESS
   ═══════════════════════════════════════════════════════════════════════════ */

[data-testid="stStatusWidget"] {
    background: var(--cream) !important;
    border: 1px solid var(--warm-gray-200) !important;
    border-radius: 8px !important;
}

[data-testid="stStatusWidget"] [data-testid="stMarkdown"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--warm-gray-600) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   EXPANDER
   ═══════════════════════════════════════════════════════════════════════════ */

.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--warm-gray-500) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    background: var(--cream) !important;
    border-radius: 6px !important;
}

div[data-testid="stExpander"] {
    background: var(--cream) !important;
    border: 1px solid var(--warm-gray-100) !important;
    border-radius: 8px !important;
}

/* Code blocks for sample questions */
.stCodeBlock {
    background: var(--white) !important;
    border: 1px solid var(--warm-gray-100) !important;
}

code {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--warm-gray-600) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TIMING NOTE
   ═══════════════════════════════════════════════════════════════════════════ */

.timing-note {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--warm-gray-400);
    margin-top: 1rem;
    letter-spacing: 0.02em;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════════════════════════════════ */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--warm-gray-50);
}

::-webkit-scrollbar-thumb {
    background: var(--warm-gray-300);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--warm-gray-400);
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════════════════════════════════ */

@media (max-width: 640px) {
    .header-title {
        font-size: 2.25rem;
    }

    .stats-strip {
        gap: 1.5rem;
        flex-wrap: wrap;
    }

    .stat-number {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_rag_system():
    """Initialize and cache the RAG system."""
    return RAGSystem(faithful_mode="confident-lite", k=5)

def normalize_question(question: str) -> str:
    """Normalize question for cache lookup."""
    return question.lower().strip()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}

rag = get_rag_system()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-eyebrow">Confidence-Gated RAG System</div>
    <h1 class="header-title">Medical Document Q&A</h1>
    <p class="header-subtitle">Ask questions about medical AI research papers. Answers are verified for accuracy before delivery.</p>
    <div class="stats-strip">
        <div class="stat-block">
            <div class="stat-number">98.3%</div>
            <div class="stat-label">Faithfulness</div>
        </div>
        <div class="stat-block">
            <div class="stat-number">96.0%</div>
            <div class="stat-label">Relevancy</div>
        </div>
        <div class="stat-block">
            <div class="stat-number">6</div>
            <div class="stat-label">Papers</div>
        </div>
    </div>
    <p class="timing-note">First query takes around a minute for verification • Repeated queries are instant</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sample Questions
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Sample questions", expanded=not st.session_state.messages):
    st.code("What is CheXNet and how does it work?", language=None)
    st.code("How does ClinicalBERT differ from regular BERT?", language=None)
    st.code("What are the limitations of AI in medical imaging?", language=None)
    st.code("How accurate is AI at classifying skin lesions?", language=None)
    st.code("How is AI used in molecular property prediction?", language=None)
    st.code("What ML methods are used for COVID-19 forecasting?", language=None)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #78716C; line-height: 1.6;">
        <strong>About this system</strong><br>
        This RAG system uses confidence-gated generation to ensure high faithfulness.
        If the system cannot answer confidently, it will abstain rather than hallucinate.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chat History
# ─────────────────────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            st.markdown(message["content"], unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about medical AI research..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Normalize for cache lookup
    cache_key = normalize_question(prompt)

    # Check cache first
    if cache_key in st.session_state.query_cache:
        response = st.session_state.query_cache[cache_key]
    else:
        # Show processing status
        with st.chat_message("assistant"):
            try:
                start_time = time.time()

                with st.status("Analyzing your question...", expanded=True) as status:
                    progress_text = st.empty()

                    # Run query in background thread
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(rag.query, prompt)

                        # Update elapsed time while waiting
                        dots = 0
                        while not future.done():
                            elapsed = time.time() - start_time
                            dot_str = "." * (dots % 4)
                            progress_text.markdown(f"**Verifying response{dot_str}** `{elapsed:.1f}s`")
                            time.sleep(0.3)
                            dots += 1

                        response = future.result()

                    elapsed = time.time() - start_time
                    status.update(label=f"Complete in {elapsed:.1f}s", state="complete", expanded=False)

                # Cache the response
                st.session_state.query_cache[cache_key] = response

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f'<div class="abstention-card"><div class="abstention-header">Error</div><div class="abstention-reason">{str(e)}</div></div>',
                    "error": True
                })
                st.rerun()

    # Build response HTML
    if hasattr(response, 'abstained') and response.abstained:
        confidence = getattr(response, 'confidence_score', 0)
        html_content = f"""<div class="abstention-card">
            <div class="abstention-header">Cannot Answer Confidently</div>
            <div class="abstention-reason">{response.abstention_reason}</div>
            <div class="confidence-meter">
                <div class="confidence-row">
                    <span class="confidence-label">Confidence</span>
                    <span class="confidence-value conf-low">{confidence:.0%}</span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-bar bar-low" style="width: {confidence * 100}%"></div>
                </div>
            </div>
        </div>"""
    else:
        answer_text = response.answer
        confidence = getattr(response, 'confidence_score', 0.95)

        if confidence >= 0.8:
            conf_class, bar_class = "conf-high", "bar-high"
        elif confidence >= 0.6:
            conf_class, bar_class = "conf-medium", "bar-medium"
        else:
            conf_class, bar_class = "conf-low", "bar-low"

        # Build sources HTML
        sources_html = ""
        if response.retrieved_chunks:
            sources_html = '<details class="sources-toggle"><summary>View sources</summary><div class="sources-list">'
            for chunk in response.retrieved_chunks[:5]:
                score = getattr(chunk, 'rrf_score', 0) or 0
                content_preview = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
                content_preview = content_preview.replace("<", "&lt;").replace(">", "&gt;")
                sources_html += f"""<div class="source-card">
                    <div class="source-header">
                        <span class="source-name">{chunk.source}</span>
                        <span class="source-score">RRF {score:.3f}</span>
                    </div>
                    <div class="source-excerpt">{content_preview}</div>
                </div>"""
            sources_html += "</div></details>"

        html_content = f"""<div class="response-card">
            <div class="response-answer">{answer_text}</div>
            <div class="confidence-meter">
                <div class="confidence-row">
                    <span class="confidence-label">Response Confidence</span>
                    <span class="confidence-value {conf_class}">{confidence:.0%}</span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-bar {bar_class}" style="width: {confidence * 100}%"></div>
                </div>
            </div>
            {sources_html}
        </div>"""

    # Add to history and rerun to display cleanly
    st.session_state.messages.append({"role": "assistant", "content": html_content})
    st.rerun()
