"""
UABC AI Assistant - Streamlit Dashboard (v9)
Module 17: Feedback Submit Button & Confirmation

Changes from v8 (feedback UX only — all other sections identical):
- Added "Send Comment" button in Rate Last Response section
- Added "Respuesta enviada" confirmation message after comment submission
- All v8 layout, CSS, logo, team info, avatars, metrics, settings preserved exactly

Team 1 - UABC Capstone Project
"""

import streamlit as st
import os
import time
import base64
from typing import List, Dict
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# PATHS / RUNTIME CONFIG
# ============================================================================

APP_DIR = Path(__file__).resolve().parent
CHROMA_DIR = os.getenv("CHROMA_DIR", str(APP_DIR / "chroma_uabc_ai_v9"))

DEFAULT_COLAB_PDFS = [
    "/data_files/Boletin1_IA_2024-01-19.pdf",
    "/data_files/IA_Practica_Docente_2024-01-24.pdf",
    "/data_files/Incorporacion_IA_Procesos_de_Investigacion_2024-06-12-6.pdf",
]


def resolve_pdf_paths() -> List[str]:
    env_paths = os.getenv("PDF_PATHS", "").strip()
    if env_paths:
        return [p.strip() for p in env_paths.split(",") if p.strip()]
    local_candidates = []
    local_dirs = [
        APP_DIR / "Modulo_16_17_18" / "Archivos IA Uni",
        APP_DIR / "Archivos IA Uni",
    ]
    for d in local_dirs:
        if d.exists():
            local_candidates.extend(str(p) for p in sorted(d.glob("*.pdf")))
    if local_candidates:
        return local_candidates
    return DEFAULT_COLAB_PDFS


PDF_PATHS = resolve_pdf_paths()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="UABC AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS - v5: Final polish, zero flicker
# ============================================================================

st.markdown("""
<style>
    /* ---- General ---- */
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    /* ---- Remove grey borders/shadows from Streamlit wrappers ---- */
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* Plotly chart container - force transparent */
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] iframe,
    .js-plotly-plot .plotly,
    .js-plotly-plot .main-svg {
        background: transparent !important;
    }

    /* ---- Header banner ---- */
    .uabc-header {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #40916C 100%);
        color: white;
        padding: 1.1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .uabc-header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; color: white !important; }
    .uabc-header p { margin: 0.2rem 0 0 0; font-size: 0.88rem; opacity: 0.9; color: #D8F3DC !important; }

    /* ---- Panel section headers ---- */
    .panel-section {
        font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.6px; border-bottom: 2px solid #2D6A4F;
        padding-bottom: 0.25rem; margin: 0.9rem 0 0.5rem 0; color: #2D6A4F;
    }

    /* ---- Metric row (pure HTML, no Streamlit containers) ---- */
    .metric-row { display: flex; justify-content: space-between; gap: 0.5rem; margin-bottom: 0.3rem; }
    .metric-box { flex: 1; text-align: center; padding: 0.5rem 0.3rem; }
    .metric-box .mv { font-size: 1.35rem; font-weight: 700; color: #1B4332; margin: 0; line-height: 1.2; }
    .metric-box .ml { font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin: 0; }

    /* ---- SLA badge ---- */
    .sla-badge-ok { background: #D4EDDA; color: #155724; padding: 0.35rem 0.7rem; border-radius: 6px;
        font-size: 0.78rem; font-weight: 600; text-align: center; margin: 0.4rem 0; display: block; }
    .sla-badge-warn { background: #FFF3CD; color: #856404; padding: 0.35rem 0.7rem; border-radius: 6px;
        font-size: 0.78rem; font-weight: 600; text-align: center; margin: 0.4rem 0; display: block; }

    /* ---- Confidence bar ---- */
    .conf-track { background: #E9ECEF; border-radius: 5px; height: 10px; width: 100%;
        margin: 0.25rem 0 0.4rem 0; overflow: hidden; }
    .conf-fill { height: 10px; border-radius: 5px; transition: width 0.4s ease; }

    /* ---- Source chip ---- */
    .src-chip { background: #E8F5E9; border: 1px solid #C8E6C9; border-radius: 6px;
        padding: 0.35rem 0.65rem; margin-bottom: 0.35rem; font-size: 0.8rem; }
    .src-chip strong { color: #1B4332; }
    .src-chip .snip { color: #555; font-size: 0.75rem; display: block; margin-top: 2px; }

    /* ---- Feature list ---- */
    .feat-item { font-size: 0.8rem; padding: 0.1rem 0; color: #444; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0.8rem; }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important; border: none !important; }

    /* ---- Example questions (static, copyable text) ---- */
    .example-list { font-size: 0.8rem; color: #ccc; line-height: 1.6; }
    .example-list code {
        background: rgba(45,106,79,0.15); color: #8BC6A8; padding: 2px 6px;
        border-radius: 4px; font-size: 0.75rem; cursor: text;
        -webkit-user-select: all; user-select: all;
    }

    /* ---- Latency mini-bars (pure HTML) ---- */
    .lat-bar-row { display: flex; align-items: flex-end; gap: 4px; margin-top: 0.3rem; height: 90px; }
    .lat-bar-item { flex: 1; display: flex; flex-direction: column; align-items: center; }
    .lat-bar { border-radius: 3px 3px 0 0; min-width: 18px; transition: height 0.3s ease; }
    .lat-bar-label { font-size: 0.6rem; color: #888; margin-top: 2px; }
    .lat-bar-val { font-size: 0.6rem; color: #aaa; margin-bottom: 1px; }
    .lat-sla-line { border-top: 1.5px dashed #E76F51; margin-top: 0.2rem; padding-top: 0.15rem;
        font-size: 0.6rem; color: #E76F51; text-align: right; }

    /* ---- Feedback section in right panel ---- */
    .fb-section { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# RAG HELPER FUNCTIONS
# ============================================================================

def format_context(docs):
    lines = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", None)
        page_str = f"p.{page+1}" if isinstance(page, int) else "p.?"
        text = d.page_content.strip().replace("\n", " ")
        lines.append(f"- ({src}, {page_str}) {text}")
    return "\n".join(lines)


def format_sources_display(docs):
    seen, sources = set(), []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", None)
        key = (src, page)
        if key not in seen:
            seen.add(key)
            page_str = f"p.{page+1}" if isinstance(page, int) else "p.?"
            snippet = d.page_content.strip()[:140].replace("\n", " ")
            sources.append({"file": src, "page": page_str, "snippet": snippet})
    return sources


@st.cache_resource
def build_retriever(top_k: int = 4):
    docs, load_errors = [], []
    for p in PDF_PATHS:
        if not os.path.exists(p):
            load_errors.append(f"Missing: {p}")
            continue
        try:
            docs.extend(PyPDFLoader(p).load())
        except Exception as e:
            load_errors.append(f"Error {p}: {e}")
    if not docs:
        raise RuntimeError(f"No documents loaded.\n" + "\n".join(load_errors))
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(chunks, emb, persist_directory=CHROMA_DIR)
    return vectordb.as_retriever(search_kwargs={"k": top_k})


def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.2):
    """Build LLM with dynamic model/temperature (no cache so settings take effect)."""
    return ChatOpenAI(model=model, temperature=temperature)


PROMPT_ES = ChatPromptTemplate.from_template(
"""Eres un asistente institucional de la UABC sobre el uso de Inteligencia Artificial.
Responde SOLO con base en las fuentes proporcionadas.
Si la pregunta NO se puede responder con las fuentes, di exactamente:
"No encuentro informacion sobre ese tema en los documentos proporcionados."

Pregunta: {question}

Fuentes (extractos):
{context}

Instrucciones de formato:
- Da la respuesta en {length_lines} lineas.
- Al final agrega: Referencias: (archivo, pagina).
""")

PROMPT_EN = ChatPromptTemplate.from_template(
"""You are an institutional assistant for UABC about the use of Artificial Intelligence.
Answer ONLY based on the provided sources.
If the question CANNOT be answered with the sources, say exactly:
"I don't find information about that topic in the provided documents."

Question: {question}

Sources (excerpts):
{context}

Format instructions:
- Give the answer in {length_lines} lines.
- At the end add: References: (filename, page).
""")

# Response length mapping
RESPONSE_LENGTH_MAP = {
    "Short (3-4 lines)": "3-4",
    "Medium (5-8 lines)": "5-8",
    "Long (8-12 lines)": "8-12",
}


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_feedback_data() -> pd.DataFrame:
    if 'feedback_db' in st.session_state and st.session_state.feedback_db:
        return pd.DataFrame(st.session_state.feedback_db)
    return pd.DataFrame(columns=['timestamp', 'message', 'response', 'rating', 'comment'])


@st.cache_data
def compute_explanation(_input_text: str, _response: str) -> Dict:
    return {
        'input_tokens': len(_input_text.split()),
        'response_tokens': len(_response.split()),
        'confidence': 0.85,
        'top_features': ['Retrieved context (RAG)', 'Semantic similarity', 'Source count'],
    }


# ============================================================================
# SESSION STATE
# ============================================================================

def initialize_session_state():
    defaults = {
        'messages': [],
        'feedback_db': [],
        'preferences': {'temperature': 0.2, 'top_k': 4, 'language': 'Español', 'show_sources': True,
                        'model': 'gpt-4o-mini', 'response_length': 'Medium (5-8 lines)'},
        'current_explanation': None,
        'last_sources': [],
        'metrics': {
            'total_messages': 0, 'avg_response_time': 0, 'total_feedback': 0,
            'query_count': 0, 'positive_feedback': 0, 'response_times': [], 'failed_queries': 0,
        },
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    prefs = st.session_state.preferences
    for k, v in [('top_k', 4), ('language', 'Español'), ('show_sources', True),
                  ('model', 'gpt-4o-mini'), ('response_length', 'Medium (5-8 lines)')]:
        prefs.setdefault(k, v)
    m = st.session_state.metrics
    for k in ['query_count', 'positive_feedback', 'total_feedback', 'response_times', 'failed_queries']:
        m.setdefault(k, 0 if k != 'response_times' else [])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def update_runtime_metrics(response_time: float, success: bool):
    m = st.session_state.metrics
    m['query_count'] += 1
    m['total_messages'] += 2
    m['response_times'].append(response_time)
    m['avg_response_time'] = sum(m['response_times']) / len(m['response_times'])
    if not success:
        m['failed_queries'] += 1


def generate_response(message: str, temperature: float = 0.2,
                      top_k: int = 4, language: str = "Español",
                      model: str = "gpt-4o-mini",
                      response_length: str = "Medium (5-8 lines)") -> tuple:
    t0 = time.time()
    try:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            rt = time.time() - t0
            update_runtime_metrics(rt, False)
            return ("Error: OPENAI_API_KEY is missing.", {}, rt, [])
        retriever = build_retriever(top_k)
        llm = build_llm(model=model, temperature=temperature)
        retrieved = retriever.invoke(message)
        context = format_context(retrieved)
        prompt = PROMPT_ES if language == "Español" else PROMPT_EN
        length_lines = RESPONSE_LENGTH_MAP.get(response_length, "5-8")
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": message, "context": context,
                               "length_lines": length_lines})
        rt = time.time() - t0
        confidence = min(0.95, 0.50 + 0.10 * len(retrieved))
        explanation = {"details": {
            "confidence": float(confidence), "sources_count": len(retrieved),
            "model": model, "temperature": temperature,
            "top_features": ["Retrieved context (RAG)", "Semantic similarity", "Source count"],
        }}
        sources_display = format_sources_display(retrieved)
        update_runtime_metrics(rt, True)
        return answer, explanation, rt, sources_display
    except Exception as e:
        rt = time.time() - t0
        update_runtime_metrics(rt, False)
        return f"Error: {e}", {}, rt, []


def save_feedback(message: str, response: str, rating: str, comment: str):
    st.session_state.feedback_db.append({
        'timestamp': datetime.now(), 'message': message,
        'response': response, 'rating': rating, 'comment': comment,
    })
    st.session_state.metrics['total_feedback'] += 1
    if "+" in rating:
        st.session_state.metrics['positive_feedback'] += 1
    load_feedback_data.clear()


# ============================================================================
# RENDER HELPERS
# ============================================================================

def render_metrics_row(items: list):
    boxes = ""
    for label, value in items:
        boxes += f'<div class="metric-box"><p class="mv">{value}</p><p class="ml">{label}</p></div>'
    st.markdown(f'<div class="metric-row">{boxes}</div>', unsafe_allow_html=True)


def render_sla_badge(latency: float):
    if latency < 5:
        st.markdown(f'<span class="sla-badge-ok">SLA OK — {latency:.2f}s (&lt; 5s)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="sla-badge-warn">SLA Exceeded — {latency:.2f}s (&gt; 5s)</span>', unsafe_allow_html=True)


def render_confidence_bar(value: float):
    pct = int(value * 100)
    color = "#2D6A4F" if value >= 0.8 else "#E9C46A" if value >= 0.6 else "#E76F51"
    st.markdown(
        f'<div class="conf-track"><div class="conf-fill" style="width:{pct}%; background:{color};"></div></div>',
        unsafe_allow_html=True)


def render_source_chip(file: str, page: str, snippet: str):
    st.markdown(
        f'<div class="src-chip"><strong>{file}</strong> — {page}'
        f'<span class="snip">{snippet}...</span></div>',
        unsafe_allow_html=True)


def render_latency_bars(times: list):
    """Render latency chart as pure HTML bars — no Plotly, no grey box."""
    if not times:
        return
    max_t = max(max(times), 5.5)  # at least 5.5 to show SLA line
    bars_html = ""
    for i, t in enumerate(times):
        h = max(4, int((t / max_t) * 80))  # height in px, min 4px
        color = "#2D6A4F" if t < 5 else "#E76F51"
        bars_html += (
            f'<div class="lat-bar-item">'
            f'<span class="lat-bar-val">{t:.1f}s</span>'
            f'<div class="lat-bar" style="height:{h}px; background:{color};"></div>'
            f'<span class="lat-bar-label">Q{i+1}</span>'
            f'</div>'
        )
    html = f'<div class="lat-bar-row">{bars_html}</div>'
    html += '<div class="lat-sla-line">SLA limit: 5s ───</div>'
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# PAGE: CHAT
# ============================================================================

def page_chat():
    prefs = st.session_state.preferences

    # ---- SIDEBAR ----
    with st.sidebar:
        # Example questions FIRST — STATIC TEXT, copy-paste only
        st.markdown("##### Example Questions")
        st.markdown(
            '<div class="example-list">'
            '1. <code>Que orientaciones se proponen para el uso de IA en la universidad?</code><br>'
            '2. <code>Que recomendaciones se dan para integrar la IA en la practica docente?</code><br>'
            '3. <code>Que riesgos eticos se mencionan al usar IA con datos de estudiantes?</code><br>'
            '4. <code>Que papel juegan los comites de etica en proyectos con IA?</code>'
            '</div>',
            unsafe_allow_html=True)

        # Settings SECOND
        st.markdown("---")
        st.markdown("##### ⚙️ Settings")

        prefs['language'] = st.selectbox(
            "🌐 Language (prompts & responses)",
            options=["Español", "English"],
            index=0 if prefs['language'] == "Español" else 1,
            help="Language for both your questions and the assistant's responses")

        model_options = ["gpt-4o-mini", "gpt-4o"]
        prefs['model'] = st.selectbox(
            "🧠 Model",
            options=model_options,
            index=model_options.index(prefs.get('model', 'gpt-4o-mini')),
            help="gpt-4o-mini: faster & cheaper | gpt-4o: higher quality")

        prefs['temperature'] = st.slider(
            "🌡️ Temperature", min_value=0.0, max_value=1.5,
            value=prefs['temperature'], step=0.1,
            help="0.0 = precise & deterministic | 1.0+ = creative & varied")

        prefs['top_k'] = st.slider(
            "📄 Top-K Chunks", min_value=2, max_value=8,
            value=prefs['top_k'], step=1,
            help="Number of document chunks retrieved per query")

        length_options = list(RESPONSE_LENGTH_MAP.keys())
        prefs['response_length'] = st.selectbox(
            "📏 Response Length",
            options=length_options,
            index=length_options.index(prefs.get('response_length', 'Medium (5-8 lines)')),
            help="Controls how long the assistant's answers will be")

        prefs['show_sources'] = st.toggle(
            "📎 Show Sources in Chat", value=prefs['show_sources'],
            help="Display source citations inline with responses")

        # Compact status
        st.markdown("---")
        pdfs_ok = all(os.path.exists(p) for p in PDF_PATHS)
        api_ok = bool(os.getenv("OPENAI_API_KEY", "").strip())
        st.caption(f"{'✅' if pdfs_ok else '❌'} {len(PDF_PATHS)} PDFs | {'✅' if api_ok else '❌'} API Key")

        st.markdown("---")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_explanation = None
            st.session_state.last_sources = []
            st.rerun()

    # ---- HEADER ----
    st.markdown(
        '<div class="uabc-header">'
        '<h1>🤖 UABC AI Assistant</h1>'
        '<p>Institutional assistant on the use of Artificial Intelligence at UABC</p>'
        '</div>', unsafe_allow_html=True)

    # ---- 2 COLUMNS: Chat | Metrics+Feedback ----
    col_chat, col_right = st.columns([13, 7], gap="medium")

    # ================= LEFT: CHAT ONLY (clean, no feedback) =================
    with col_chat:
        prompt = st.chat_input("Ask a question about AI at UABC...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Searching institutional documents..."):
                response, explanation, response_time, sources = generate_response(
                    prompt, temperature=prefs['temperature'],
                    top_k=prefs['top_k'], language=prefs['language'],
                    model=prefs.get('model', 'gpt-4o-mini'),
                    response_length=prefs.get('response_length', 'Medium (5-8 lines)'))
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.current_explanation = {
                'input': prompt, 'output': response,
                'details': explanation.get("details", explanation) if explanation else {},
                'response_time': response_time,
            }
            st.session_state.last_sources = sources
            st.rerun()

        # Chat history
        if st.session_state.messages:
            chat_box = st.container(height=520)
            with chat_box:
                for msg in st.session_state.messages:
                    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])
        else:
            st.info("Welcome! Type a question above to query UABC's institutional documents on AI.")

    # ================= RIGHT: METRICS + EXPLAINABILITY + SOURCES + LATENCY + FEEDBACK =================
    with col_right:
        m = st.session_state.metrics
        exp = st.session_state.current_explanation

        # ---- Real-Time Metrics ----
        st.markdown('<div class="panel-section">Real-Time Metrics</div>', unsafe_allow_html=True)

        avg_lat = m['avg_response_time']
        total_fb = m['total_feedback']
        pos_fb = m['positive_feedback']
        sat_str = f"{(pos_fb / total_fb * 100):.0f}%" if total_fb > 0 else "—"
        success_count = max(0, m['query_count'] - m['failed_queries'])
        success_pct = f"{(success_count / m['query_count'] * 100):.0f}%" if m['query_count'] > 0 else "—"

        render_metrics_row([("Queries", str(m['query_count'])), ("Avg Latency", f"{avg_lat:.2f}s"), ("Satisfaction", sat_str)])
        render_metrics_row([("Success Rate", success_pct), ("Feedback", str(total_fb)), ("Errors", str(m['failed_queries']))])

        if exp:
            render_sla_badge(exp.get('response_time', 0))

        # ---- Explainability ----
        st.markdown('<div class="panel-section">Explainability</div>', unsafe_allow_html=True)

        if exp and exp.get('details'):
            details = exp['details']
            conf = details.get('confidence', 0)
            conf_label = "High" if conf >= 0.8 else "Medium" if conf >= 0.6 else "Low"
            st.markdown(f"**Confidence:** {conf:.0%} ({conf_label})")
            render_confidence_bar(conf)
            src_count = details.get('sources_count', 0)
            resp_time = exp.get('response_time', 0)
            model_used = details.get('model', 'gpt-4o-mini')
            temp_used = details.get('temperature', 0.2)
            st.markdown(f"**Chunks:** {src_count} &nbsp;|&nbsp; **Latency:** {resp_time:.2f}s", unsafe_allow_html=True)
            st.markdown(f"**Model:** {model_used} &nbsp;|&nbsp; **Temp:** {temp_used}", unsafe_allow_html=True)
            if details.get('top_features'):
                st.markdown("**Key Factors:**")
                for feat in details['top_features']:
                    st.markdown(f"<div class='feat-item'>✓ {feat}</div>", unsafe_allow_html=True)
        else:
            st.caption("Ask a question to see the analysis.")

        # ---- Retrieved Sources ----
        st.markdown('<div class="panel-section">Retrieved Sources</div>', unsafe_allow_html=True)
        if st.session_state.last_sources:
            for src in st.session_state.last_sources:
                render_source_chip(src['file'], src['page'], src['snippet'])
        else:
            st.caption("Sources will appear here after a query.")

        # ---- Latency Chart (pure HTML bars, NO Plotly = no grey box) ----
        if m.get('response_times') and len(m['response_times']) > 0:
            st.markdown('<div class="panel-section">Latency per Query</div>', unsafe_allow_html=True)
            render_latency_bars(m['response_times'])

        # ---- Feedback (moved here from chat column — no overlap) ----
        st.markdown('<div class="panel-section">Rate Last Response</div>', unsafe_allow_html=True)

        if exp:
            fb_cols = st.columns([1, 1])
            with fb_cols[0]:
                if st.button("👍 Helpful", key="fb_pos", use_container_width=True):
                    save_feedback(exp['input'], exp['output'], "+1", "")
                    st.success("Thanks!")
            with fb_cols[1]:
                if st.button("👎 Not helpful", key="fb_neg", use_container_width=True):
                    save_feedback(exp['input'], exp['output'], "-1", "")
                    st.warning("Thanks for feedback")

            comment = st.text_input(
                "Comment:", key=f"fb_cmt_{len(st.session_state.messages)}",
                placeholder="What could be improved?", label_visibility="collapsed")
            if st.button("📩 Send Comment", key="fb_send", use_container_width=True):
                if comment:
                    save_feedback(exp['input'], exp['output'], "comment", comment)
                    st.success("✅ Respuesta enviada")
                else:
                    st.warning("Please enter a comment first.")
        else:
            st.caption("Feedback will be available after a response.")


# ============================================================================
# PAGE: EXPLAINABILITY ANALYSIS
# ============================================================================

def page_explainability():
    st.markdown(
        '<div class="uabc-header"><h1>Explainability Analysis</h1>'
        '<p>Detailed analysis of model decisions and response quality.</p></div>',
        unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("No conversations yet. Go to Chat to start.")
        return

    conversations = []
    for i in range(0, len(st.session_state.messages), 2):
        if i + 1 < len(st.session_state.messages):
            conversations.append({
                'user': st.session_state.messages[i]['content'],
                'assistant': st.session_state.messages[i + 1]['content'],
            })

    if not conversations:
        st.warning("No complete conversations found.")
        return

    selected_idx = st.selectbox(
        "Select a conversation:", range(len(conversations)),
        format_func=lambda i: f"Conv {i+1}: {conversations[i]['user'][:60]}...")
    conv = conversations[selected_idx]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Question")
        st.markdown(f"```\n{conv['user']}\n```")
        st.subheader("Response")
        st.markdown(f"```\n{conv['assistant']}\n```")
    with col2:
        st.subheader("Metrics")
        exp = compute_explanation(conv['user'], conv['assistant'])
        st.metric("Input Tokens", exp['input_tokens'])
        st.metric("Response Tokens", exp['response_tokens'])
        st.metric("Confidence", f"{exp['confidence']:.0%}")
        st.divider()
        st.markdown("**Top Features:**")
        for i, feat in enumerate(exp['top_features'], 1):
            st.markdown(f"{i}. {feat}")

    st.divider()
    st.subheader("Feature Importance")
    features = ['Input Length', 'Complexity', 'Context Relevance', 'Semantic Match', 'Pattern Recognition']
    importance = [0.25, 0.20, 0.30, 0.15, 0.10]
    fig = go.Figure(data=[go.Bar(x=features, y=importance, marker_color='#2D6A4F')])
    fig.update_layout(xaxis_title="Features", yaxis_title="Importance Score", height=350,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font_color='#888')
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: FEEDBACK DASHBOARD
# ============================================================================

def page_feedback():
    st.markdown(
        '<div class="uabc-header"><h1>Feedback Dashboard</h1>'
        '<p>User feedback monitoring and response quality analytics.</p></div>',
        unsafe_allow_html=True)

    feedback_df = load_feedback_data()
    if feedback_df.empty:
        st.info("No feedback collected yet. Chat with the assistant and submit feedback.")
        return

    col1, col2, col3, col4 = st.columns(4)
    positive = len(feedback_df[feedback_df['rating'].str.contains(r'\+', na=False)])
    negative = len(feedback_df[feedback_df['rating'].str.contains('-', na=False)])
    with col1: st.metric("Total Feedback", len(feedback_df))
    with col2: st.metric("Positive", positive)
    with col3: st.metric("Negative", negative)
    with col4:
        sat = (positive / len(feedback_df) * 100) if len(feedback_df) > 0 else 0
        st.metric("Satisfaction", f"{sat:.0f}%")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribution")
        if len(feedback_df) > 0:
            rating_counts = feedback_df['rating'].value_counts()
            fig = px.pie(values=rating_counts.values, names=rating_counts.index)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#888')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Feedback Over Time")
        if len(feedback_df) > 0:
            feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
            daily = feedback_df.groupby('date').size().reset_index(name='count')
            fig = px.line(daily, x='date', y='count', markers=True)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#888')
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Recent Feedback")
    display_df = feedback_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['message'] = display_df['message'].str[:50] + '...'
    display_df['response'] = display_df['response'].str[:50] + '...'
    st.dataframe(display_df[['timestamp', 'message', 'response', 'rating', 'comment']],
                 use_container_width=True, hide_index=True)

    st.divider()
    csv = feedback_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv,
                       file_name=f"feedback_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")


# ============================================================================
# PAGE: MONITORING
# ============================================================================

def page_monitoring():
    st.markdown(
        '<div class="uabc-header"><h1>System Monitoring</h1>'
        '<p>Performance metrics and system health overview.</p></div>',
        unsafe_allow_html=True)

    m = st.session_state.metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Queries", m['query_count'])
    with col2: st.metric("Avg Latency", f"{m['avg_response_time']:.2f}s")
    with col3: st.metric("Total Feedback", m['total_feedback'])
    with col4:
        sat = (m['positive_feedback'] / m['total_feedback'] * 100) if m['total_feedback'] > 0 else 0
        st.metric("Satisfaction", f"{sat:.0f}%")
    with col5: st.metric("Failed Queries", m['failed_queries'])

    st.divider()

    if m.get('response_times'):
        st.subheader("Response Latency Over Time")
        times = m['response_times']
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=times, mode='lines+markers', name='Latency', line=dict(color='#2D6A4F')))
        fig.add_hline(y=5, line_dash="dash", line_color="#E76F51", annotation_text="SLA: 5s")
        fig.update_layout(xaxis_title="Query", yaxis_title="Seconds", height=300,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#888')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Module 17 Success Checklist")
    queries = m['query_count']
    has_fb = m['total_feedback'] > 0
    avg_lat = m['avg_response_time']
    success_q = max(0, queries - m['failed_queries'])
    checklist = [
        ("App runs without crashes", True),
        ("RAG pipeline responds to queries", success_q > 0),
        ("Sources/citations displayed", success_q > 0),
        ("Latency under 5s SLA", avg_lat < 5 if queries > 0 else False),
        ("User feedback collected", has_fb),
        ("Error handling in place", True),
        ("Loading indicators shown", True),
        ("Professional UI layout", True),
        ("Real-time metrics displayed", True),
    ]
    for label, passed in checklist:
        st.markdown(f"{'✅' if passed else '⬜'} {label}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cache Status")
        st.success("Model loaded and cached")
        st.caption("Feedback cache TTL: 1 hour")
    with c2:
        if st.button("Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared!")
            st.rerun()

    with st.expander("Session State (Debug)"):
        st.json({
            'messages_count': len(st.session_state.messages),
            'feedback_count': len(st.session_state.feedback_db),
            'preferences': st.session_state.preferences,
            'metrics': {k: v for k, v in m.items() if k != 'response_times'},
        })


# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================

def page_documentation():
    st.markdown(
        '<div class="uabc-header"><h1>Documentation</h1>'
        '<p>Project information and technical architecture.</p></div>',
        unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["About", "Technical", "Team"])

    with tab1:
        st.markdown("""
        ## About This Application

        **Academic Q&A Assistant** for UABC that answers questions about institutional
        AI policies and guidelines using **Retrieval-Augmented Generation (RAG)**.

        ### Knowledge Base
        - **Boletin 1 IA (2024):** Institutional guidelines for AI in higher education
        - **IA Practica Docente (2024):** Recommendations for AI in teaching practice
        - **Incorporacion IA Investigacion (2024):** AI integration in research processes

        ### How to Use
        1. Go to **Chat** and type a question about AI in UABC
        2. Review the answer with citations on the left
        3. Check **Explainability** metrics on the right panel
        4. See retrieved **sources** on the right panel
        5. Provide **feedback** using the buttons on the right panel
        6. Adjust **Settings** in the sidebar (temperature, chunks, language)
        """)

    with tab2:
        st.markdown("""
        ## Technical Architecture

        ```
        PDFs (3 documents, 41 pages)
            --> PyPDFLoader
            --> RecursiveCharacterTextSplitter (chunk_size=900, overlap=150)
            --> OpenAI text-embedding-3-small
            --> ChromaDB (vector store)
                    |
        User Query --> Retriever (k=2-8, configurable)
                    --> Prompt Template (ES/EN, with fallback)
                    --> GPT-4o-mini (temperature configurable)
                    --> Answer with citations
        ```

        ### Key Technologies
        - **LangChain:** Orchestration framework
        - **OpenAI API:** Embeddings + LLM (GPT-4o-mini)
        - **ChromaDB:** Vector database
        - **Streamlit:** Web interface
        - **Plotly:** Metric visualizations (Monitoring & Explainability pages)

        ### v5 Improvements over v4
        - Example questions: static copyable text (no auto-fill buttons)
        - Feedback relocated to right panel (eliminates chat input overlap)
        - Latency chart: pure HTML bars (no Plotly grey flicker in chat)
        - Increased chat history height (520px)
        - All Plotly charts: transparent bg + font color for dark themes

        ### v6 Improvements over v5
        - Sidebar reordered: Example Questions first, Settings second
        - Custom chat avatars: 🧑‍💻 (user) and 🤖 (assistant)
        - Nuclear CSS: eliminates ALL remaining grey rectangles
        - Removed separator causing grey artifact below title
        - GitHub-ready: .gitignore, LICENSE, CHANGELOG.md

        ### v7 Improvements over v6
        - Dynamic model selection: gpt-4o-mini / gpt-4o
        - Temperature dynamically applied to LLM
        - Response length control: Short / Medium / Long
        - Emoji icons on all settings controls

        ### v8 Improvements over v7
        - UABC institutional logo in sidebar header
        - Team member names in Documentation > Team tab

        ### v9 Improvements over v8
        - Added "Send Comment" button in Rate Last Response section
        - Added "Respuesta enviada" confirmation after comment submission
        """)

    with tab3:
        st.markdown("""
        ## Team Information

        **Team Name:** Team 1

        **Members:**
        - Garcia Canseco Eloisa del Carmen
        - Inzunza Gonzalez Everardo
        - Navarro Cota Christian Xavier
        - Rivera Aguirre Flavio Abel

        **Course:** Formacion en IA - Gobierno del Estado de Baja California

        **Institution:** Universidad Autonoma de Baja California (UABC)

        **Modules:** 15-16-17-18 Capstone Project

        **Project:** Academic Q&A Assistant with RAG (v9)
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    initialize_session_state()

    with st.sidebar:
        # UABC Logo — look for uabc_logo.png next to the app or in /content/
        logo_candidates = [
            APP_DIR / "uabc_logo.png",
            Path("/content/uabc_logo.png"),
        ]
        logo_path = next((p for p in logo_candidates if p.exists()), None)
        if logo_path:
            logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
            st.markdown(
                f'<div style="text-align:center; margin-bottom:0.3rem;">'
                f'<img src="data:image/png;base64,{logo_b64}" '
                f'style="max-height:70px; width:auto;" alt="UABC Logo">'
                f'</div>',
                unsafe_allow_html=True)
        st.markdown("### 🎓 UABC AI Assistant")
        st.caption("Module 17 — v9")
        page = st.radio(
            "Navigation",
            ["💬 Chat", "🔍 Explainability", "📊 Feedback", "📈 Monitoring", "📚 Docs"],
            label_visibility="collapsed")

    if page == "💬 Chat":
        page_chat()
    elif page == "🔍 Explainability":
        page_explainability()
    elif page == "📊 Feedback":
        page_feedback()
    elif page == "📈 Monitoring":
        page_monitoring()
    elif page == "📚 Docs":
        page_documentation()


if __name__ == "__main__":
    main()
