# app.py

import streamlit as st
import plotly.graph_objects as go
import json
from text_generator import generate_text
import modules.bias_detector as bias_detector
import modules.bias_detector1 as bias_detector1
from llm_analyzer import analyze_bias_with_llm
from modules.bias_mitigator import mitigate_bias
from file_processor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_docx
)

st.set_page_config(
    page_title="Bias Lens",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400;1,700&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:       #1a1410;
    --paper:     #f5f0e8;
    --cream:     #ede8dc;
    --warm-mid:  #c8bfaa;
    --rule:      #2a2018;
    --red:       #c0392b;
    --amber:     #d4860a;
    --green:     #2d6a4f;
    --blue:      #1a3a5c;
    --muted:     #6b6050;
    --highlight: #fff3cd;
}

* { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Source Serif 4', Georgia, serif !important;
    background-color: var(--paper) !important;
    color: var(--ink) !important;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1100px !important; }

/* ── MASTHEAD ── */
.masthead {
    border-top: 4px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    padding: 1.2rem 0 1rem 0;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    color: var(--ink);
    line-height: 1;
}
.masthead-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-align: right;
    line-height: 1.6;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.masthead-rule {
    border: none;
    border-top: 3px double var(--rule);
    margin: 0.4rem 0 1.5rem 0;
}

/* ── DECK / SUBTITLE ── */
.deck {
    font-family: 'Source Serif 4', serif;
    font-size: 1.05rem;
    font-style: italic;
    font-weight: 300;
    color: var(--muted);
    margin-bottom: 1.8rem;
    border-left: 3px solid var(--warm-mid);
    padding-left: 0.8rem;
    line-height: 1.5;
}

/* ── INPUT AREA ── */
.input-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
div[data-testid="stTextInput"] input {
    background: white !important;
    border: 1.5px solid var(--rule) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 1rem !important;
    padding: 0.6rem 0.9rem !important;
    box-shadow: 2px 2px 0 var(--warm-mid) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--red) !important;
    box-shadow: 2px 2px 0 var(--red) !important;
    outline: none !important;
}
div[data-testid="stButton"] button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
div[data-testid="stButton"] button:hover {
    background: var(--red) !important;
}

/* ── SECTION RULES ── */
.section-rule {
    border: none;
    border-top: 2px solid var(--ink);
    margin: 1.8rem 0 0.6rem 0;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    margin-bottom: 0.8rem;
}
.section-headline {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.6rem;
    line-height: 1.2;
}

/* ── GENERATED TEXT ── */
.article-body {
    font-family: 'Source Serif 4', serif;
    font-size: 0.97rem;
    line-height: 1.8;
    color: var(--ink);
    background: white;
    border: 1px solid var(--warm-mid);
    padding: 1.4rem 1.6rem;
    border-left: 4px solid var(--ink);
    box-shadow: 3px 3px 0 var(--cream);
    margin-bottom: 1rem;
}

/* ── VERDICT STRIP ── */
.verdict-strip {
    display: flex;
    gap: 0;
    border: 1.5px solid var(--ink);
    margin: 1.4rem 0;
    box-shadow: 3px 3px 0 var(--warm-mid);
}
.verdict-cell {
    flex: 1;
    padding: 0.9rem 1rem;
    border-right: 1px solid var(--warm-mid);
    background: white;
    text-align: center;
}
.verdict-cell:last-child { border-right: none; }
.verdict-num {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
    color: var(--ink);
}
.verdict-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* ── SEVERITY PILL ── */
.sev-none   { display:inline-block;padding:.2rem .7rem;font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:500;letter-spacing:.06em;text-transform:uppercase;background:#e8f5e9;color:var(--green);border:1.5px solid var(--green); }
.sev-low    { display:inline-block;padding:.2rem .7rem;font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:500;letter-spacing:.06em;text-transform:uppercase;background:#fff8e1;color:var(--amber);border:1.5px solid var(--amber); }
.sev-medium { display:inline-block;padding:.2rem .7rem;font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:500;letter-spacing:.06em;text-transform:uppercase;background:#fff3e0;color:#e65100;border:1.5px solid #e65100; }
.sev-high   { display:inline-block;padding:.2rem .7rem;font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:500;letter-spacing:.06em;text-transform:uppercase;background:#ffebee;color:var(--red);border:1.5px solid var(--red); }

/* ── NO BIAS ── */
.clean-verdict {
    background: #f0f7f4;
    border: 1.5px solid var(--green);
    border-left: 5px solid var(--green);
    padding: 1.2rem 1.5rem;
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-style: italic;
    color: var(--green);
    margin: 1rem 0;
}

/* ── LAYER HEADERS ── */
.layer-head {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.6rem 0 0.4rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--warm-mid);
}
.layer-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1;
}
.layer-title {
    font-family: 'Source Serif 4', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
}
.layer-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.15rem 0.5rem;
    border: 1px solid var(--warm-mid);
    color: var(--muted);
    background: var(--cream);
}
.layer-desc {
    font-family: 'Source Serif 4', serif;
    font-size: 0.82rem;
    font-style: italic;
    color: var(--muted);
    margin-bottom: 0.8rem;
    line-height: 1.5;
}

/* ── FINDING CARDS ── */
.finding {
    background: white;
    border: 1px solid var(--warm-mid);
    border-left: 4px solid var(--ink);
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 2px 2px 0 var(--cream);
}
.finding-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.4rem;
}
.finding-type {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    background: var(--cream);
    border: 1px solid var(--warm-mid);
    padding: 0.1rem 0.5rem;
}
.finding-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.3rem;
    font-style: italic;
}
.finding-quote {
    font-family: 'Source Serif 4', serif;
    font-size: 0.85rem;
    font-style: italic;
    color: var(--blue);
    background: #f0f4f8;
    border-left: 3px solid var(--blue);
    padding: 0.4rem 0.7rem;
    margin: 0.4rem 0;
    line-height: 1.5;
}
.finding-body {
    font-family: 'Source Serif 4', serif;
    font-size: 0.87rem;
    color: #3a3028;
    line-height: 1.6;
    margin-top: 0.4rem;
}
.finding-context {
    font-family: 'Source Serif 4', serif;
    font-size: 0.78rem;
    font-style: italic;
    color: var(--muted);
    margin-top: 0.5rem;
    border-top: 1px solid var(--cream);
    padding-top: 0.4rem;
}

/* ── ASSESSMENT BOX ── */
.assessment {
    background: var(--highlight);
    border: 1px solid #e6c84a;
    border-left: 4px solid var(--amber);
    padding: 0.9rem 1.2rem;
    font-family: 'Source Serif 4', serif;
    font-size: 0.9rem;
    font-style: italic;
    color: var(--ink);
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* ── EMPTY STATE ── */
.empty-layer {
    font-family: 'Source Serif 4', serif;
    font-size: 0.88rem;
    font-style: italic;
    color: var(--muted);
    padding: 0.7rem 1rem;
    background: var(--cream);
    border: 1px solid var(--warm-mid);
}

/* ── REASONING LOG ── */
.log-row {
    display: flex;
    gap: 0.8rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--cream);
    font-size: 0.83rem;
    line-height: 1.5;
}
.log-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--warm-mid);
    min-width: 1.5rem;
    padding-top: 0.1rem;
}
.log-text { color: var(--muted); font-family: 'Source Serif 4', serif; font-style: italic; }

/* ── FOOTER ── */
.footer-rule {
    border: none;
    border-top: 3px double var(--rule);
    margin: 3rem 0 0.6rem 0;
}
.footer-txt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--warm-mid);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: center;
}

/* Plotly transparent bg */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── MITIGATED TEXT ── */
.mitigated-body {
    font-family: 'Source Serif 4', serif;
    font-size: 0.97rem;
    line-height: 1.8;
    color: var(--ink);
    background: #f0f7f4;
    border: 1px solid #2d6a4f;
    padding: 1.4rem 1.6rem;
    border-left: 4px solid var(--green);
    box-shadow: 3px 3px 0 #c8e6c9;
    margin-bottom: 1rem;
}

/* ── CHANGE ITEM ── */
.change-item {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid var(--cream);
    font-size: 0.82rem;
    line-height: 1.5;
    background: white;
}
.change-arrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--green);
    min-width: 1rem;
    padding-top: 0.15rem;
}
.change-text {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    color: var(--muted);
}

/* ── STRATEGY BADGE ── */
.strategy-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.15rem 0.6rem;
    border: 1px solid var(--green);
    color: var(--green);
    background: #f0f7f4;
    margin-left: 0.5rem;
}

/* ── COMPARISON COLUMNS ── */
.compare-col-head {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

/* ── TAB STYLING ── */
[data-testid="stTabs"] [role="tab"]:nth-of-type(1) {
    color: var(--red) !important;
    transition: color 0.15s !important;
}
[data-testid="stTabs"] [role="tab"]:nth-of-type(1):hover {
    color: #8c291f !important;
}

[data-testid="stTabs"] [role="tab"]:nth-of-type(2) {
    color: var(--blue) !important;
    transition: color 0.15s !important;
}
[data-testid="stTabs"] [role="tab"]:nth-of-type(2):hover {
    color: #0f2440 !important;
}

/* ── RADIO BUTTON STYLING ── */
[data-testid="stRadio"] label:nth-of-type(1) {
    color: var(--red) !important;
    transition: color 0.15s !important;
}
[data-testid="stRadio"] label:nth-of-type(1):hover {
    color: #8c291f !important;
}

[data-testid="stRadio"] label:nth-of-type(2) {
    color: var(--blue) !important;
    transition: color 0.15s !important;
}
[data-testid="stRadio"] label:nth-of-type(2):hover {
    color: #0f2440 !important;
}
</style>
""", unsafe_allow_html=True)

# ── MASTHEAD ────────────────────────────────────────────────
import datetime
today = datetime.date.today().strftime("%B %d, %Y").upper()

st.markdown(f"""
<div class="masthead">
    <div class="masthead-title">Bias Lens</div>
    <div class="masthead-meta">
        Comprehensive Multi-Category Bias Detection<br>
        7 Bias Types · Dual-Layer Analysis System<br>
        {today}
    </div>
</div>
<hr class="masthead-rule">
<div class="deck">
    An investigative tool that generates text from any prompt and examines it for 7 types of bias —
    using ML classifiers alongside AI contextual reasoning.
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE INIT ───────────────────────────────────────
if "locked_text" not in st.session_state:
    st.session_state.locked_text = ""
if "locked_prompt" not in st.session_state:
    st.session_state.locked_prompt = ""
if "test_cases" not in st.session_state:
    st.session_state.test_cases = []
if "test_case_counter" not in st.session_state:
    st.session_state.test_case_counter = 0

# Initialize button states
run       = False
reanalyze = False

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Analyze", "📋 Test Cases"])

def sev_class(sev):
    return {"None":"sev-none","Low":"sev-low","Medium":"sev-medium","High":"sev-high"}.get(sev,"sev-none")

# ── TAB 1: ANALYZE ───────────────────────────────────────────
# ── CREATE PERSISTENT INPUT CONTAINER ─────────────────────────
input_container = st.container()
results_container = st.container()

with input_container:
    with tab1:

        st.subheader("Choose Input Source")

        input_mode = st.radio(
            "Select Input Type",
            [
                "Generate Biased Text",
                "Upload Document",
                "Manual Text"
            ]
        )

        # ── Model mode toggle ─────────────────────────────────────
        st.markdown('<div class="input-label">Layer 1 Model Mode</div>', unsafe_allow_html=True)
        model_mode = st.radio(
            label="model_mode",
            options=["RoBERTa + ModernBERT (Full)", "RoBERTa Only (Ablation)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        if model_mode == "RoBERTa Only (Ablation)":
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                        background:#fff8e1;border:1px solid #d4860a;border-left:3px solid #d4860a;
                        padding:.5rem .8rem;color:#d4860a;margin-bottom:.8rem;">
                ⚠ ABLATION MODE — ModernBERT disabled. Bias type classification limited to pattern matching only.
                ML-detected sentences without pattern matches will show as "Potential Bias" instead of Gender/Age Bias.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                        background:#f0f7f4;border:1px solid #2d6a4f;border-left:3px solid #2d6a4f;
                        padding:.5rem .8rem;color:#2d6a4f;margin-bottom:.8rem;">
                ✓ FULL MODE — RoBERTa detects bias, ModernBERT classifies type (Gender / Age / Non-Binary).
            </div>""", unsafe_allow_html=True)

        # ── INPUT SOURCE HANDLING ─────────────────────
uploaded_text = ""
manual_text = ""
final_text = ""

with input_container:
    if input_mode == "Upload Document":

        uploaded_file = st.file_uploader(
            "Upload a PDF, TXT, or DOCX file",
            type=["pdf", "txt", "docx"]
        )

        if uploaded_file is not None:

            file_name = uploaded_file.name.lower()

            try:
                if file_name.endswith(".pdf"):
                    uploaded_text = extract_text_from_pdf(uploaded_file)

                elif file_name.endswith(".txt"):
                    uploaded_text = extract_text_from_txt(uploaded_file)

                elif file_name.endswith(".docx"):
                    uploaded_text = extract_text_from_docx(uploaded_file)

                st.success("Document uploaded successfully!")
                st.info(f"""
File Name: {uploaded_file.name}

Size: {uploaded_file.size/1024:.2f} KB

Characters Extracted: {len(uploaded_text)}
""")

                st.markdown(
                    f'<div class="article-body">{uploaded_text[:2000]}</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.exception(e)

    elif input_mode == "Manual Text":

        manual_text = st.text_area(
            "Paste your text here",
            height=250,
            placeholder="Enter text to analyze for bias..."
        )

    elif input_mode == "Generate Biased Text":

        st.markdown(
            '<div class="input-label">Enter a topic or question to analyze</div>',
            unsafe_allow_html=True
        )

prompt = ""

with input_container:
    if input_mode == "Generate Biased Text":

        col_input, col_btn = st.columns([5, 1])

        with col_input:
            prompt = st.text_input(
                label="prompt",
                placeholder='e.g. "Why do men make better leaders than women?"',
                label_visibility="collapsed"
            )

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Run Analysis"):
                run = True

    else:

        if st.button("Run Analysis"):
            run = True

        # ── Re-analyze button (uses locked text, no regeneration) ──
        if st.session_state.locked_text:
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                        color:var(--muted);margin-top:.3rem;">
                📌 Locked text available from prompt:
                <strong>"{st.session_state.locked_prompt[:60]}{'...' if len(st.session_state.locked_prompt) > 60 else ''}"</strong>
            </div>""", unsafe_allow_html=True)
            if st.button("🔁 Re-analyze same text with current model mode",
                         help="Same text, different model — for fair comparison"):
                reanalyze = True

# ── MAIN LOGIC ───────────────────────────────────────────────

if run or reanalyze:

    # ── Step 1: Get text ──────────────────────────────────────
    if reanalyze and st.session_state.locked_text:
        generated_text = st.session_state.locked_text
        prompt         = st.session_state.locked_prompt
        with input_container:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                        background:#f0f4f8;border:1px solid #1a3a5c;border-left:3px solid #1a3a5c;
                        padding:.5rem .8rem;color:#1a3a5c;margin-bottom:.5rem;">
                🔁 Using locked text — no new text generated. Same input, different model mode.
            </div>""", unsafe_allow_html=True)
    else:

        # ── Generated Text Mode ─────────────────
        if input_mode == "Generate Biased Text":

            if not prompt.strip():
                st.warning("Please enter a prompt.")
                st.stop()

            with st.spinner("Generating text…"):
                try:
                    generated_text = generate_text(prompt)

                    st.session_state.locked_text = generated_text
                    st.session_state.locked_prompt = prompt

                except Exception as e:
                    st.error(f"Text generation failed: {e}")
                    st.stop()

        # ── Upload Mode ─────────────────────────
        elif input_mode == "Upload Document":

            if not uploaded_text.strip():
                st.warning("Please upload a document.")
                st.stop()

            generated_text = uploaded_text[:15000]

        # ── Manual Text Mode ────────────────────
        elif input_mode == "Manual Text":

            if not manual_text.strip():
                st.warning("Please enter text.")
                st.stop()

            generated_text = manual_text
            st.session_state.locked_text   = generated_text
            st.session_state.locked_prompt = prompt

    if not generated_text:
        st.error("The model returned empty text. Try a different prompt.")
        st.stop()
    # ── Step 2: Run Layer 1 ───────────────────────────────────
    with st.spinner("Running ML bias classifier…"):
        try:
            if model_mode == "RoBERTa Only (Ablation)":
                rule_result = bias_detector1.detect_bias(generated_text)
            else:
                rule_result = bias_detector.detect_bias(generated_text)
        except Exception as e:
            st.error(f"Rule-based detection failed: {e}")
            st.stop()

    # ── Step 3: Run Layer 2 ───────────────────────────────────
    with st.spinner("Running AI contextual analysis…"):
        try:
            llm_result = analyze_bias_with_llm(generated_text)
        except Exception as e:
            llm_result = {"biases_found":[],"overall_assessment":f"Unavailable: {e}","overall_severity":"None"}

    # ── Step 3b: Run mitigation ───────────────────────────────
    with st.spinner("Generating mitigated version…"):
        mitigation = mitigate_bias(generated_text, rule_result, llm_result)

    # ── Step 4: Display results ───────────────────────────────
    with results_container:
        with tab1:
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Generated Text</div>', unsafe_allow_html=True)
        source_type = input_mode
        st.markdown(f"**Source:** {source_type}")
        st.markdown(f'<div class="article-body">{generated_text}</div>', unsafe_allow_html=True)

        # ── VERDICT STRIP ────────────────────────────────────
        all_bias_types = list(set(
            rule_result["bias_types"] +
            [b["bias_type"] for b in llm_result.get("biases_found",[])]
        ))
        sev_order = {"None":0,"Low":1,"Medium":2,"High":3}
        combined_sev = max(
            rule_result["severity"],
            llm_result.get("overall_severity","None"),
            key=lambda s: sev_order.get(s,0)
        )
        total_llm = len(llm_result.get("biases_found",[]))
        total_bias_pct = rule_result.get("total_bias_percentage", 0)

        st.markdown(f"""
        <div class="verdict-strip">
            <div class="verdict-cell">
                <div class="verdict-num">{rule_result["bias_score"]}</div>
                <div class="verdict-lbl">ML Score</div>
            </div>
            <div class="verdict-cell">
                <div class="verdict-num">{total_bias_pct:.1f}%</div>
                <div class="verdict-lbl">Total Bias %</div>
            </div>
            <div class="verdict-cell">
                <div class="verdict-num">{total_llm}</div>
                <div class="verdict-lbl">AI Findings</div>
            </div>
            <div class="verdict-cell">
                <div class="verdict-num" style="font-size:1.1rem;padding-top:.5rem;">
                    <span class="{sev_class(combined_sev)}">{combined_sev}</span>
                </div>
                <div class="verdict-lbl">Overall Severity</div>
            </div>
            <div class="verdict-cell">
                <div class="verdict-num">{len(all_bias_types)}</div>
                <div class="verdict-lbl">Bias Types</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── NO BIAS ───────────────────────────────────────────
        if not rule_result["bias_detected"] and total_llm == 0:
            st.markdown("""
            <div class="clean-verdict">
                ✓ &nbsp; No significant gender or age bias detected in this text.
            </div>""", unsafe_allow_html=True)

        else:
            # ── CHARTS ───────────────────────────────────────
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Visual Summary</div>', unsafe_allow_html=True)

            ch_left, ch_right = st.columns(2)

            with ch_left:
                # Calculate bias percentages from Layer 1 counts
                bias_counts = rule_result.get("bias_counts", {})
                total_biases = sum(bias_counts.values())
                
                if total_biases > 0:
                    # Only include biases with count > 0
                    detected_biases = {k: v for k, v in bias_counts.items() if v > 0}
                    bias_labels = []
                    bias_values = []
                    
                    for bias_key, count in detected_biases.items():
                        bias_name = {
                            "gender": "Gender",
                            "occupation": "Occupation",
                            "religion_caste": "Religion/Caste",
                            "nationality": "Nationality",
                            "adversarial": "Adversarial",
                            "representation": "Representation",
                            "sentiment_groups": "Sentiment Toward Groups"
                        }.get(bias_key, bias_key.title())
                        
                        percentage = (count / total_biases) * 100
                        bias_labels.append(f"{bias_name}<br>({percentage:.1f}%)")
                        bias_values.append(count)
                    
                    colors = ["#1a3a5c","#c0392b","#d4860a","#2d6a4f","#6b3fa0","#7a5230","#c44536"]
                    fig_pie = go.Figure(go.Pie(
                        labels=bias_labels,
                        values=bias_values,
                        hole=0.4,
                        marker=dict(colors=colors[:len(bias_labels)],
                                    line=dict(color="#f5f0e8", width=2)),
                        textfont=dict(color="white", size=9, family="JetBrains Mono"),
                        hovertemplate="%{label}: %{value} detected<extra></extra>"
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#6b6050", family="Source Serif 4"),
                        margin=dict(l=10,r=10,t=30,b=10), height=280,
                        title=dict(text="Bias Breakdown (Layer 1)", font=dict(family="Playfair Display", size=13, color="#1a1410"), x=0),
                        legend=dict(font=dict(color="#3a3028",size=9,family="Source Serif 4"), bgcolor="rgba(0,0,0,0)", x=0, y=-0.1)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.markdown('<div style="text-align:center;padding:2rem;color:#6b6050;">No biases detected</div>', unsafe_allow_html=True)

            with ch_right:
                gauge_color = {"None":"#2d6a4f","Low":"#d4860a","Medium":"#e65100","High":"#c0392b"}.get(combined_sev,"#1a3a5c")
                max_score = max(rule_result["bias_score"], 10)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rule_result["bias_score"],
                    number={"font":{"color":"#1a1410","family":"Playfair Display","size":30}},
                    title={"text":"ML Bias Score","font":{"family":"Playfair Display","size":13,"color":"#1a1410"}},
                    gauge={
                        "axis":{"range":[0,max_score+2],"tickwidth":1,"tickcolor":"#c8bfaa",
                                "tickfont":{"color":"#6b6050","size":9,"family":"JetBrains Mono"}},
                        "bar":{"color":gauge_color,"thickness":0.3},
                        "bgcolor":"white","borderwidth":1,"bordercolor":"#c8bfaa",
                        "steps":[
                            {"range":[0,2],"color":"#f0f7f4"},
                            {"range":[2,5],"color":"#fff8e1"},
                            {"range":[5,max_score+2],"color":"#fff0f0"}
                        ],
                        "threshold":{"line":{"color":gauge_color,"width":2},"thickness":0.8,"value":rule_result["bias_score"]}
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20,r=20,t=40,b=10), height=230
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ── NEW PROPORTIONAL METRICS DISPLAY ─────────────
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Proportional Bias Methodology (v2.0)</div>', unsafe_allow_html=True)
            
            # Display key metrics in an organized grid
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.markdown(f"""
                <div style="background:#f5f0e8;border:1px solid #c8bfaa;border-left:4px solid #1a1410;padding:1rem;border-radius:2px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;text-transform:uppercase;color:#6b6050;margin-bottom:.4rem;">A) Total Bias Metrics</div>
                    <div style="font-family:'Playfair Display',serif;font-size:.95rem;font-weight:700;color:#1a1410;margin-bottom:.3rem;">
                        {rule_result.get('biased_sentences', 0)} of {rule_result.get('total_sentences', 0)} sentences
                    </div>
                    <div style="font-family:'Source Serif 4',serif;font-size:.82rem;color:#6b6050;">
                        <strong>{total_bias_pct:.2f}%</strong> of text contains bias
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div style="background:#f5f0e8;border:1px solid #c8bfaa;border-left:4px solid #c0392b;padding:1rem;border-radius:2px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;text-transform:uppercase;color:#6b6050;margin-bottom:.4rem;">B) Bias Score</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:900;color:#1a1410;">
                        {rule_result["bias_score"]}
                    </div>
                    <div style="font-family:'Source Serif 4',serif;font-size:.75rem;color:#6b6050;line-height:1.4;">
                        Formula: ({total_bias_pct:.2f}% ÷ 20) × 2<br>= {rule_result["bias_score"]} points
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                st.markdown(f"""
                <div style="background:#f5f0e8;border:1px solid #c8bfaa;border-left:4px solid #2d6a4f;padding:1rem;border-radius:2px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;text-transform:uppercase;color:#6b6050;margin-bottom:.4rem;">Confidence</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;color:#1a1410;">
                        {rule_result.get('confidence_avg', 0):.1%}
                    </div>
                    <div style="font-family:'Source Serif 4',serif;font-size:.75rem;color:#6b6050;">
                        Avg detection confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display category distribution and contribution
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">C) Category Distribution & Actual Contribution</div>', unsafe_allow_html=True)
            
            # Get metrics data
            proportional_dist = rule_result.get("proportional_distribution", {})
            actual_contrib = rule_result.get("actual_contribution", {})
            
            # Filter and sort by contribution (descending)
            category_metrics = {}
            for cat, contribution in actual_contrib.items():
                if contribution > 0.01:  # Only show non-negligible contributions
                    category_metrics[cat] = {
                        "count": rule_result.get("bias_counts", {}).get(cat, 0),
                        "proportional": proportional_dist.get(cat, 0),
                        "contribution": contribution
                    }
            
            if category_metrics:
                # Sort by contribution descending
                sorted_cats = sorted(category_metrics.items(), key=lambda x: x[1]["contribution"], reverse=True)
                
                # Display as rows
                for cat, metrics in sorted_cats:
                    cat_display = {
                        "gender": "Gender Bias",
                        "occupation": "Occupation Bias",
                        "religion_caste": "Religion/Caste Bias",
                        "nationality": "Nationality Bias",
                        "adversarial": "Adversarial/Toxicity",
                        "representation": "Representation",
                        "sentiment_groups": "Sentiment Toward Groups"
                    }.get(cat, cat.title())
                    
                    st.markdown(f"""
                    <div style="background:white;border:1px solid #c8bfaa;padding:.8rem 1rem;margin-bottom:.5rem;border-radius:2px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;">
                            <div style="font-family:'Playfair Display',serif;font-size:.95rem;font-weight:700;color:#1a1410;">
                                {cat_display}
                            </div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#6b6050;">
                                {metrics['count']} instances
                            </div>
                        </div>
                        <div style="display:flex;gap:1.5rem;margin-top:.6rem;">
                            <div>
                                <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;text-transform:uppercase;color:#6b6050;margin-bottom:.3rem;">Proportional (within detected bias)</div>
                                <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;color:#1a1410;">
                                    {metrics['proportional']:.2f}%
                                </div>
                            </div>
                            <div>
                                <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;text-transform:uppercase;color:#6b6050;margin-bottom:.3rem;">Actual Contribution (to text)</div>
                                <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;color:#c0392b;">
                                    {metrics['contribution']:.2f}%
                                </div>
                            </div>
                        </div>
                        <div style="font-family:'Source Serif 4',serif;font-size:.75rem;color:#6b6050;margin-top:.5rem;">
                            <em>Calculation: ({total_bias_pct:.2f}% total × {metrics['proportional']:.2f}% share) = {metrics['contribution']:.2f}%</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align:center;padding:1rem;color:#6b6050;">No bias categories detected</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="layer-head">
                <div class="layer-num">I</div>
                <div>
                    <div class="layer-title">ML Classifier &nbsp;
                        <span class="layer-tag">{"RoBERTa Only — Ablation" if model_mode == "RoBERTa Only (Ablation)" else "RoBERTa + ModernBERT"}</span>
                    </div>
                </div>
            </div>
            <div class="layer-desc">
                {"Sentence-level detection using RoBERTa only. Bias type determined by pattern matching — ModernBERT type classifier is disabled for comparison." if model_mode == "RoBERTa Only (Ablation)" else "Sentence-level detection using two fine-tuned transformer models. himel7/bias-detector (92% accuracy) flags biased sentences; cirimus/modernbert classifies gender vs. age type."}
            </div>
            """, unsafe_allow_html=True)

            if not rule_result["bias_detected"]:
                st.markdown('<div class="empty-layer">No explicit bias patterns flagged by the ML classifier.</div>', unsafe_allow_html=True)
            else:
                for e in rule_result["evidence"]:
                    conf = e.get("confidence","")
                    conf_html = f'<span style="font-family:JetBrains Mono,monospace;font-size:.65rem;color:var(--muted);">{conf}% confidence</span>' if conf else ""
                    sent_html = f'<div class="finding-context">Sentence: "{e.get("sentence","")}"</div>' if e.get("sentence") else ""
                    st.markdown(f"""
                    <div class="finding">
                        <div class="finding-top">
                            <span class="finding-type">{e.get("type","Bias")}</span>
                            {conf_html}
                        </div>
                        <div class="finding-quote">{e.get("text","")}</div>
                        <div class="finding-body">{e.get("explanation","")}</div>
                        {sent_html}
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="section-label" style="margin-top:1rem;">Classifier Log</div>', unsafe_allow_html=True)
                for i, r in enumerate(rule_result["reasons"],1):
                    st.markdown(f'<div class="log-row"><span class="log-num">{i:02d}</span><span class="log-text">{r}</span></div>', unsafe_allow_html=True)

            # ── LAYER 2 ───────────────────────────────────────
            st.markdown("""
            <div class="layer-head">
                <div class="layer-num">II</div>
                <div>
                    <div class="layer-title">AI Contextual Analysis &nbsp;<span class="layer-tag">Llama 3.3 70B</span></div>
                </div>
            </div>
            <div class="layer-desc">
                Deep contextual reasoning across 7 bias categories: gender, occupation, religion/caste, nationality, 
                adversarial/toxicity, representation imbalance, and sentiment toward groups.
            </div>
            """, unsafe_allow_html=True)

            # ── LAYER 2 CHARTS ─────────────────────────────────
            if llm_result.get("biases_found"):
                st.markdown('<div class="section-label">Visual Summary (Layer 2)</div>', unsafe_allow_html=True)
                
                ch2_left, ch2_right = st.columns(2)
                
                with ch2_left:
                    # Calculate bias type distribution for Layer 2
                    llm_biases = llm_result.get("biases_found", [])
                    llm_bias_counts = {}
                    for bias in llm_biases:
                        bias_type = bias.get("bias_type", "Unknown")
                        llm_bias_counts[bias_type] = llm_bias_counts.get(bias_type, 0) + 1
                    
                    if llm_bias_counts:
                        llm_labels = []
                        llm_values = []
                        total_llm = sum(llm_bias_counts.values())
                        
                        for bias_type, count in llm_bias_counts.items():
                            percentage = (count / total_llm) * 100
                            llm_labels.append(f"{bias_type}<br>({percentage:.1f}%)")
                            llm_values.append(count)
                        
                        colors = ["#1a3a5c","#c0392b","#d4860a","#2d6a4f","#6b3fa0","#7a5230","#c44536"]
                        fig_pie_llm = go.Figure(go.Pie(
                            labels=llm_labels,
                            values=llm_values,
                            hole=0.4,
                            marker=dict(colors=colors[:len(llm_labels)],
                                        line=dict(color="#f5f0e8", width=2)),
                            textfont=dict(color="white", size=9, family="JetBrains Mono"),
                            hovertemplate="%{label}: %{value} found<extra></extra>"
                        ))
                        fig_pie_llm.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#6b6050", family="Source Serif 4"),
                            margin=dict(l=10,r=10,t=30,b=10), height=280,
                            title=dict(text="Bias Breakdown (Layer 2)", font=dict(family="Playfair Display", size=13, color="#1a1410"), x=0),
                            legend=dict(font=dict(color="#3a3028",size=9,family="Source Serif 4"), bgcolor="rgba(0,0,0,0)", x=0, y=-0.1)
                        )
                        st.plotly_chart(fig_pie_llm, use_container_width=True)
                
                with ch2_right:
                    # Layer 2 score (count of biases)
                    llm_score = len(llm_result.get("biases_found", []))
                    llm_severity = llm_result.get("overall_severity", "Low")
                    
                    gauge_color_llm = {"None":"#2d6a4f","Low":"#d4860a","Medium":"#e65100","High":"#c0392b"}.get(llm_severity,"#1a3a5c")
                    max_llm_score = max(llm_score, 10)
                    
                    fig_gauge_llm = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=llm_score,
                        number={"font":{"color":"#1a1410","family":"Playfair Display","size":30}},
                        title={"text":"AI Bias Count","font":{"family":"Playfair Display","size":13,"color":"#1a1410"}},
                        gauge={
                            "axis":{"range":[0,max_llm_score+2],"tickwidth":1,"tickcolor":"#c8bfaa",
                                    "tickfont":{"color":"#6b6050","size":9,"family":"JetBrains Mono"}},
                            "bar":{"color":gauge_color_llm,"thickness":0.3},
                            "bgcolor":"white","borderwidth":1,"bordercolor":"#c8bfaa",
                            "steps":[
                                {"range":[0,2],"color":"#f0f7f4"},
                                {"range":[2,5],"color":"#fff8e1"},
                                {"range":[5,max_llm_score+2],"color":"#fff0f0"}
                            ],
                            "threshold":{"line":{"color":gauge_color_llm,"width":2},"thickness":0.8,"value":llm_score}
                        }
                    ))
                    fig_gauge_llm.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20,r=20,t=40,b=10), height=280
                    )
                    st.plotly_chart(fig_gauge_llm, use_container_width=True)

            if llm_result.get("overall_assessment"):
                st.markdown(f'<div class="assessment">"{llm_result["overall_assessment"]}"</div>', unsafe_allow_html=True)

            if not llm_result.get("biases_found"):
                st.markdown('<div class="empty-layer">No subtle contextual biases detected by AI analysis.</div>', unsafe_allow_html=True)
            else:
                sev_colors = {"Low":("#fff8e1","#d4860a"),"Medium":("#fff3e0","#e65100"),"High":("#ffebee","#c0392b"),"None":("#f0f7f4","#2d6a4f")}
                for b in llm_result["biases_found"]:
                    b_sev = b.get("severity","Low")
                    bg, fg = sev_colors.get(b_sev,("#fff8e1","#d4860a"))
                    st.markdown(f"""
                    <div class="finding" style="border-left-color:{fg};">
                        <div class="finding-top">
                            <span class="finding-type">{b.get("bias_type","Bias")}</span>
                            <span style="font-family:JetBrains Mono,monospace;font-size:.65rem;
                                         background:{bg};color:{fg};padding:.1rem .5rem;
                                         border:1px solid {fg};">{b_sev}</span>
                        </div>
                        <div class="finding-title">{b.get("title","")}</div>
                        <div class="finding-quote">{b.get("evidence","")}</div>
                        <div class="finding-body">{b.get("explanation","")}</div>
                    </div>""", unsafe_allow_html=True)

            # ── LAYER III — 7-STAGE ADVANCED NLP MITIGATION ─────
            st.markdown("""
            <div class="layer-head">
                <div class="layer-num">III</div>
                <div class="layer-title">Bias Mitigation</div>
            </div>
            """, unsafe_allow_html=True)

            if not mitigation["bias_reduced"]:
                st.markdown(
                    f'<div class="empty-layer">{mitigation["summary"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="assessment">{mitigation["summary"]}</div>',
                    unsafe_allow_html=True
                )

                # ── Before / After ────────────────────────────────
                st.markdown('<div class="section-label">Before / After — Final Result</div>', unsafe_allow_html=True)
                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown('<div class="compare-col-head">Original — Biased</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="article-body">{mitigation["original_text"]}</div>', unsafe_allow_html=True)
                with right_col:
                    st.markdown('<div class="compare-col-head">Mitigated — Neutral</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="mitigated-body">{mitigation["final_text"]}</div>', unsafe_allow_html=True)
                    st.download_button(
    "Download Mitigated Text",
    mitigation["final_text"],
    file_name="mitigated_text.txt"
)

                # ── Validation Metrics ────────────────────────────
                val = mitigation.get("validation", {})
                if val.get("meaning_similarity") is not None:
                    vm1, vm2, vm3, vm4 = st.columns(4)
                    vm1.metric("Meaning Preserved",
                               f'{val["meaning_similarity"]:.2%}',
                               delta="✓ OK" if val["meaning_preserved"] else "⚠ Rolled back")
                    vm2.metric("Overall Bias Reduced",
                               f'{val.get("overall_bias_reduction","N/A")}%')
                    vm3.metric("Neutrality Score",
                               f'{val.get("neutrality_score", 0):.3f}')
                    vm4.metric("Gender / Age Sensitivity",
                               f'{val.get("mitig_gender_sensitivity",0):.3f} / {val.get("mitig_age_sensitivity",0):.3f}',
                               delta=f'was {val.get("orig_gender_sensitivity",0):.3f} / {val.get("orig_age_sensitivity",0):.3f}')
                    if val.get("warning"):
                        st.warning(val["warning"])

    # ── AUTO-SAVE TEST CASE ──────────────────────────────────────────────────────
    # Save this analysis as a test case
    st.session_state.test_case_counter += 1
    test_case = {
        "id": st.session_state.test_case_counter,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": st.session_state.locked_prompt if st.session_state.locked_prompt else "N/A",
        "generated_text": generated_text[:500] if generated_text else "",
        "layer1": {
            "bias_score": rule_result.get("bias_score", 0),
            "bias_detected": rule_result.get("bias_detected", False),
            "total_bias_pct": total_bias_pct,
            "evidence": rule_result.get("evidence", [])[:3]  # Store first 3 findings
        },
        "layer2": {
            "findings_count": total_llm,
            "overall_assessment": llm_result.get("overall_assessment", "N/A"),
            "overall_severity": llm_result.get("overall_severity", "None")
        },
        "layer3": {
            "bias_reduced": mitigation.get("bias_reduced", False),
            "summary": mitigation.get("summary", "N/A")
        },
        "combined_severity": combined_sev,
        "bias_types": all_bias_types
    }
    st.session_state.test_cases.append(test_case)

# ── TAB 2: TEST CASES ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Test Case Statistics</div>', unsafe_allow_html=True)

    if not st.session_state.test_cases:
        st.markdown('<div class="empty-layer">No test cases saved yet. Run some analyses in the Analyze tab first.</div>', unsafe_allow_html=True)
    else:
        # ── Calculate statistics ──────────────────────────────
        total_cases = len(st.session_state.test_cases)
        biased_cases = sum(1 for tc in st.session_state.test_cases if tc["layer1"]["bias_detected"])
        clean_cases = total_cases - biased_cases
        high_sev = sum(1 for tc in st.session_state.test_cases if tc["combined_severity"] == "High")
        med_sev = sum(1 for tc in st.session_state.test_cases if tc["combined_severity"] == "Medium")
        
        # ── STATS ROW ──────────────────────────────────────────
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Cases", total_cases)
        s2.metric("Biased Cases", biased_cases)
        s3.metric("Clean Cases", clean_cases)
        s4.metric("High Severity", high_sev)
        s5.metric("Medium Severity", med_sev)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── SEVERITY DISTRIBUTION ─────────────────────────────
        st.markdown('<div class="section-label">Severity Distribution</div>', unsafe_allow_html=True)
        sev_dist = {
            "None": sum(1 for tc in st.session_state.test_cases if tc["combined_severity"] == "None"),
            "Low": sum(1 for tc in st.session_state.test_cases if tc["combined_severity"] == "Low"),
            "Medium": med_sev,
            "High": high_sev
        }
        sd1, sd2, sd3, sd4 = st.columns(4)
        colors_sev = {"None": "#2d6a4f", "Low": "#d4860a", "Medium": "#e65100", "High": "#c0392b"}
        for col, (sev, count) in zip([sd1, sd2, sd3, sd4], sev_dist.items()):
            col.markdown(f'<div style="text-align:center;padding:.6rem;border:1.5px solid {colors_sev[sev]};background:white;"><div style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:900;color:{colors_sev[sev]};">{count}</div><div style="font-family:JetBrains Mono,monospace;font-size:.6rem;text-transform:uppercase;color:{colors_sev[sev]};">{sev}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── BIAS TYPE FREQUENCY ───────────────────────────────
        st.markdown('<div class="section-label">Most Common Bias Types</div>', unsafe_allow_html=True)
        bias_type_freq = {}
        for tc in st.session_state.test_cases:
            for bias_type in tc["bias_types"]:
                bias_type_freq[bias_type] = bias_type_freq.get(bias_type, 0) + 1
        
        if bias_type_freq:
            sorted_biases = sorted(bias_type_freq.items(), key=lambda x: x[1], reverse=True)[:6]
            for btype, count in sorted_biases:
                pct = min(round((count / total_cases) * 100), 100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:.4rem;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;
                                color:var(--muted);min-width:220px;">{btype}</div>
                    <div style="flex:1;background:var(--cream);height:8px;border-radius:0;">
                        <div style="width:{pct}%;background:var(--blue);height:8px;"></div>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;
                                color:var(--ink);min-width:40px;">{count}x</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── EXPORT / CLEAR BUTTONS ────────────────────────────
        col_exp, col_clr, _ = st.columns([2, 2, 6])
        with col_exp:
            json_str = json.dumps(st.session_state.test_cases, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇ Export JSON",
                data=json_str,
                file_name="bias_lens_test_cases.json",
                mime="application/json"
            )
        with col_clr:
            if st.button("🗑 Clear All Test Cases"):
                st.session_state.test_cases = []
                st.session_state.test_case_counter = 0
                st.rerun()
        
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">All Test Cases</div>', unsafe_allow_html=True)
        
        # ── INDIVIDUAL CASES ──────────────────────────────────
        sev_colors = {
            "None":   ("#f0f7f4", "#2d6a4f"),
            "Low":    ("#fff8e1", "#d4860a"),
            "Medium": ("#fff3e0", "#e65100"),
            "High":   ("#ffebee", "#c0392b"),
        }
        
        for idx, case in enumerate(reversed(st.session_state.test_cases)):
            sev = case["combined_severity"]
            bg, fg = sev_colors.get(sev, ("#fff8e1", "#d4860a"))
            types_str = ", ".join(case["bias_types"]) if case["bias_types"] else "None"
            
            with st.expander(f'#{case["id"]}  |  {case["timestamp"]}  |  Severity: {sev}  |  {case["prompt"][:60]}{"..." if len(case["prompt"]) > 60 else ""}'):
                
                # Prompt + Text
                st.markdown(f'<div class="section-label">Prompt</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="article-body" style="margin-bottom:.5rem;">{case["prompt"]}</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="section-label">Generated Text (Preview)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="article-body">{case["generated_text"]}</div>', unsafe_allow_html=True)
                
                # Summary strip
                st.markdown(f"""
                <div style="display:flex;gap:0;border:1.5px solid var(--ink);margin:.8rem 0;box-shadow:2px 2px 0 var(--warm-mid);">
                    <div style="flex:1;padding:.6rem;text-align:center;background:white;border-right:1px solid var(--warm-mid);">
                        <div style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:900;color:var(--ink);">{case['layer1']['bias_score']}</div>
                        <div style="font-family:JetBrains Mono,monospace;font-size:.6rem;text-transform:uppercase;color:var(--muted);">ML Score</div>
                    </div>
                    <div style="flex:1;padding:.6rem;text-align:center;background:white;border-right:1px solid var(--warm-mid);">
                        <div style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:900;color:var(--ink);">{case['layer2']['findings_count']}</div>
                        <div style="font-family:JetBrains Mono,monospace;font-size:.6rem;text-transform:uppercase;color:var(--muted);">AI Findings</div>
                    </div>
                    <div style="flex:1;padding:.6rem;text-align:center;background:{bg};">
                        <div style="font-family:JetBrains Mono,monospace;font-size:.85rem;font-weight:700;color:{fg};padding-top:.3rem;">{sev}</div>
                        <div style="font-family:JetBrains Mono,monospace;font-size:.6rem;text-transform:uppercase;color:{fg};">Severity</div>
                    </div>
                    <div style="flex:2;padding:.6rem;text-align:center;background:white;">
                        <div style="font-family:'Source Serif 4',serif;font-size:.8rem;color:var(--ink);">{types_str}</div>
                        <div style="font-family:JetBrains Mono,monospace;font-size:.6rem;text-transform:uppercase;color:var(--muted);">Bias Types</div>
                    </div>
                </div>""", unsafe_allow_html=True)
                
                # Layer 1 findings
                if case["layer1"]["evidence"]:
                    st.markdown('<div class="section-label">Layer 1 — Sample Findings</div>', unsafe_allow_html=True)
                    for e in case["layer1"]["evidence"][:2]:
                        st.markdown(f"""
                        <div class="finding">
                            <span class="finding-type">{e.get('type', 'Bias')}</span>
                            <div class="finding-quote">{e.get('text', '')}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="empty-layer">Layer 1: No ML findings.</div>', unsafe_allow_html=True)
                
                # Layer 2 findings
                st.markdown('<div class="section-label">Layer 2 — AI Summary</div>', unsafe_allow_html=True)
                if case["layer2"]["overall_assessment"] and case["layer2"]["overall_assessment"] != "N/A":
                    st.markdown(f'<div class="assessment">"{case["layer2"]["overall_assessment"][:500]}"</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="empty-layer">No detailed assessment available.</div>', unsafe_allow_html=True)
                
                # Layer 3 summary
                st.markdown('<div class="section-label">Layer 3 — Mitigation</div>', unsafe_allow_html=True)
                if case["layer3"]["bias_reduced"]:
                    st.markdown(f'<div class="assessment">✓ {case["layer3"]["summary"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="empty-layer">{case["layer3"]["summary"]}</div>', unsafe_allow_html=True)
                
                # Delete button
                if st.button(f"🗑 Delete case #{case['id']}", key=f"del_{idx}_{case['id']}"):
                    # Remove this case and rebuild
                    st.session_state.test_cases = [tc for tc in st.session_state.test_cases if tc["id"] != case["id"]]
                    st.rerun()

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<hr class="footer-rule">
<div class="footer-txt">
    Bias Lens &nbsp;·&nbsp; Gender, Non-Binary & Age Bias Detection And Mitigation
    &nbsp;·&nbsp; RoBERTa + ModernBERT + Llama 3.3 70B &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)