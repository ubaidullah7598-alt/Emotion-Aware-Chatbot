"""
app.py  –  EmoSense RAG  (HuggingFace Edition · 100% Free)
────────────────────────────────────────────────────────────
Run:   streamlit run app.py
"""

import streamlit as st
from rag_system    import RAGSystem
from emotion_graph import build_graph

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Emotion metadata ─────────────────────────────────────────────────────
EMOTION_META = {
    "happy":      {"emoji": "😊", "color": "#FFD93D", "label": "Happy"},
    "excited":    {"emoji": "🤩", "color": "#FF6B6B", "label": "Excited"},
    "neutral":    {"emoji": "😐", "color": "#A8DADC", "label": "Neutral"},
    "confused":   {"emoji": "😕", "color": "#C77DFF", "label": "Confused"},
    "frustrated": {"emoji": "😤", "color": "#FF9F43", "label": "Frustrated"},
    "sad":        {"emoji": "😢", "color": "#74B9FF", "label": "Sad"},
    "angry":      {"emoji": "😠", "color": "#FF4757", "label": "Angry"},
    "anxious":    {"emoji": "😰", "color": "#FFA502", "label": "Anxious"},
    "stressed":   {"emoji": "😫", "color": "#FF6348", "label": "Stressed"},
}

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.stApp { background: #0d0d14; color: #e8e8f0; }

[data-testid="stSidebar"]   { background: #12121c !important; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }

.user-bubble {
    background: linear-gradient(135deg, #0d7377, #14a085);
    color: white; padding: 12px 18px;
    border-radius: 20px 20px 4px 20px; max-width: 75%;
    margin-left: auto; margin-bottom: 6px;
    font-size: 15px; line-height: 1.6;
    box-shadow: 0 4px 20px rgba(13,115,119,0.35);
}
.assistant-bubble {
    background: #1a1a28; border: 1px solid #2a2a3e;
    color: #e0e0f0; padding: 12px 18px;
    border-radius: 20px 20px 20px 4px; max-width: 76%;
    margin-bottom: 6px; font-size: 15px; line-height: 1.6;
}
.emotion-chip {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 6px;
}
.chat-row-user      { display: flex; flex-direction: row-reverse; margin-bottom: 14px; }
.chat-row-assistant { display: flex; flex-direction: row;         margin-bottom: 14px; }
.avatar {
    width: 38px; height: 38px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0; margin: 0 8px;
}
.stButton > button {
    background: linear-gradient(135deg, #0d7377, #14a085);
    color: white; border: none; border-radius: 12px;
    font-weight: 600; padding: 10px 28px;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

.stTextArea textarea {
    background: #1a1a28 !important; color: #e8e8f0 !important;
    border: 1px solid #2e2e45 !important; border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 15px !important;
}
.stTextArea textarea:focus {
    border-color: #0d7377 !important;
    box-shadow: 0 0 0 2px rgba(13,115,119,.25) !important;
}
[data-testid="stFileUploader"] {
    border: 1px dashed #2e2e45 !important;
    border-radius: 14px !important; background: #12121c !important;
}
[data-testid="stMetric"]      { background:#1a1a28; border-radius:12px; padding:12px 16px; border:1px solid #2a2a3e; }
[data-testid="stMetricValue"] { color:#e8e8f0 !important; }
[data-testid="stMetricLabel"] { color:#888 !important; }
.section-title { font-family:'DM Serif Display',serif; font-size:28px; color:#e8e8f0; margin-bottom:2px; }
.section-sub   { color:#555; font-size:12px; letter-spacing:2px; text-transform:uppercase; margin-bottom:18px; }
hr { border-color:#1e1e2e !important; }
[data-testid="stExpander"] { background:#1a1a28 !important; border:1px solid #2a2a3e !important; border-radius:12px !important; }
.hf-badge {
    display:inline-flex; align-items:center; gap:8px;
    background:#0d1a1a; border:1px solid #0d737744;
    border-radius:999px; padding:6px 16px;
    color:#0d9e9e; font-size:12px; font-weight:700; letter-spacing:1px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────
def _init():
    defaults = {
        "chat_history":    [],
        "rag":             RAGSystem(),
        "graph":           None,
        "current_emotion": "neutral",
        "hf_token":        "",
        "model_name":      "mistralai/Mistral-7B-Instruct-v0.3",
        "ingest_log":      [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

def get_graph():
    if st.session_state.graph is None:
        st.session_state.graph = build_graph(st.session_state.rag)
    return st.session_state.graph


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px;'>
        <div style='font-size:46px; margin-bottom:6px;'>🧠</div>
        <div style='font-family:"DM Serif Display",serif; font-size:22px; color:#e8e8f0;'>
            EmoSense RAG
        </div>
        <div style='color:#444; font-size:11px; letter-spacing:2px; text-transform:uppercase; margin-top:4px;'>
            Emotion-Aware · LangGraph · RAG
        </div>
        <div class='hf-badge' style='margin:10px auto 0; display:inline-flex;'>
            🤗 Powered by HuggingFace
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── HF Token ──────────────────────────────────────────────────────────
    st.markdown("### 🔑 HuggingFace Token")
    st.markdown("""
    <div style='font-size:12px; color:#666; margin-bottom:8px;'>
        Free at <a href='https://huggingface.co/settings/tokens'
        style='color:#0d9e9e;' target='_blank'>huggingface.co/settings/tokens</a>
        → New token → Role: <b>Read</b>
    </div>
    """, unsafe_allow_html=True)

    token = st.text_input(
        "HF Token",
        value=st.session_state.hf_token,
        type="password",
        placeholder="hf_...",
        label_visibility="collapsed",
    )
    st.session_state.hf_token = token

    if token.startswith("hf_"):
        st.success("✅ Token looks good!")
    elif token:
        st.warning("⚠️ Token should start with hf_")
    else:
        st.info("Paste your free HuggingFace token above.")

    st.divider()

    # ── Model selection ───────────────────────────────────────────────────
    st.markdown("### 🤗 Model")
    MODELS = {
        "mistralai/Mistral-7B-Instruct-v0.3":         "Mistral 7B Instruct  ✅ recommended",
        "HuggingFaceH4/zephyr-7b-beta":               "Zephyr 7B Beta       ✅ great chat",
        "tiiuae/falcon-7b-instruct":                  "Falcon 7B Instruct   ✅ reliable",
        "google/flan-t5-large":                       "Flan-T5 Large        ✅ very fast",
        "mistralai/Mixtral-8x7B-Instruct-v0.1":       "Mixtral 8x7B         ✅ most capable",
    }
    selected = st.selectbox(
        "Model",
        list(MODELS.keys()),
        format_func=lambda x: MODELS[x],
        index=list(MODELS.keys()).index(st.session_state.model_name)
              if st.session_state.model_name in MODELS else 0,
        label_visibility="collapsed",
    )
    if selected != st.session_state.model_name:
        st.session_state.model_name = selected
        st.session_state.graph = None

    with st.expander("ℹ️ Model notes"):
        st.markdown("""
- **Mistral 7B** — best balance, works immediately ✅  
- **Zephyr 7B** — fine-tuned for chat, very good ✅  
- **Falcon 7B** — reliable fallback ✅  
- **Flan-T5 Large** — tiny & fast, less conversational ✅  
- **Mixtral 8x7B** — most powerful, slightly slower ✅  
        """)

    st.divider()

    # ── Document Upload ───────────────────────────────────────────────────
    st.markdown("### 📚 Knowledge Base (RAG)")
    st.caption("Upload PDFs or text files for RAG-powered answers.")

    uploads = st.file_uploader(
        "Drop files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploads:
        for f in uploads:
            already = any(f.name in log for log in st.session_state.ingest_log)
            if not already:
                with st.spinner(f"Indexing {f.name}…"):
                    msg = st.session_state.rag.add_document(f)
                    st.session_state.ingest_log.append(msg)
                    st.session_state.graph = None

    for log in st.session_state.ingest_log:
        st.markdown(f"<div style='font-size:12px;color:#888;'>{log}</div>",
                    unsafe_allow_html=True)

    if st.session_state.rag.has_documents:
        st.success(f"📖 {len(st.session_state.rag.loaded_files)} file(s) loaded")
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.rag.clear()
            st.session_state.ingest_log = []
            st.session_state.graph = None
            st.rerun()
    else:
        st.info("No documents — model answers from training data.")

    st.divider()

    if st.button("🔄 Clear conversation", use_container_width=True):
        st.session_state.chat_history  = []
        st.session_state.current_emotion = "neutral"
        st.rerun()

    with st.expander("🔍 LangGraph Pipeline"):
        st.markdown("""
```
START
  │
  ▼
detect_emotion
  HF model → JSON emotion
  │
  ▼
retrieve_context
  FAISS similarity search
  (skipped if no docs)
  │
  ▼
generate_response
  HF model → emotion-aware
  + RAG-grounded reply
  │
  ▼
END
```""")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN – Header
# ══════════════════════════════════════════════════════════════════════════
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<div class="section-title">EmoSense RAG Assistant</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">HuggingFace · LangChain · LangGraph · FAISS · Streamlit</div>',
                unsafe_allow_html=True)
with col_badge:
    emo = EMOTION_META.get(st.session_state.current_emotion, EMOTION_META["neutral"])
    st.markdown(f"""
    <div style='text-align:right; padding-top:8px;'>
        <div style='display:inline-block; background:rgba(255,255,255,.05);
                    border:1px solid {emo["color"]}55; border-radius:999px; padding:8px 18px;'>
            <span style='font-size:22px;'>{emo["emoji"]}</span>
            <span style='color:{emo["color"]}; font-size:12px; font-weight:700;
                         letter-spacing:1px; text-transform:uppercase; margin-left:6px;'>
                {emo["label"]}
            </span>
        </div>
    </div>""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("💬 Messages",  len(st.session_state.chat_history))
m2.metric("📚 Documents", len(st.session_state.rag.loaded_files))
m3.metric("🤗 Model",     st.session_state.model_name.split("/")[-1][:16])
m4.metric("🔍 RAG",       "Active" if st.session_state.rag.has_documents else "Off")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
#  CHAT DISPLAY
# ══════════════════════════════════════════════════════════════════════════
if not st.session_state.chat_history:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#444;'>
        <div style='font-size:52px; margin-bottom:12px;'>💬</div>
        <div style='font-size:18px; color:#666;'>Start a conversation</div>
        <div style='font-size:13px; color:#444; margin-top:6px;'>
            I'll detect your emotion and answer using your uploaded documents if loaded.
        </div>
    </div>""", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    role  = msg["role"]
    emo_i = EMOTION_META.get(msg.get("emotion","neutral"), EMOTION_META["neutral"])
    conf  = msg.get("confidence","")

    if role == "user":
        st.markdown(f"""
        <div class="chat-row-user">
            <div class="avatar" style="background:#0d737722;">👤</div>
            <div><div class="user-bubble">{msg["content"]}</div></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row-assistant">
            <div class="avatar"
                 style="background:{emo_i['color']}22;
                        box-shadow:0 0 12px {emo_i['color']}44;">
                {emo_i['emoji']}
            </div>
            <div>
                <div class="emotion-chip"
                     style="background:{emo_i['color']}22;
                            color:{emo_i['color']};
                            border:1px solid {emo_i['color']}44;">
                    {emo_i['label']} · {conf}
                </div>
                <div class="assistant-bubble">{msg["content"]}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  INPUT AREA
# ══════════════════════════════════════════════════════════════════════════
st.divider()

inp_col, btn_col = st.columns([5, 1])
with inp_col:
    user_text = st.text_area(
        "Message",
        placeholder="Share what's on your mind…",
        height=80,
        key="user_input",
        label_visibility="collapsed",
    )
with btn_col:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    send_btn = st.button("Send ➤", use_container_width=True, type="primary")


# ── Process ───────────────────────────────────────────────────────────────
if send_btn and user_text.strip():

    if not st.session_state.hf_token:
        st.error("⚠️  Please enter your HuggingFace token in the sidebar.")
        st.stop()

    graph = get_graph()

    initial_state = {
        "user_input":         user_text.strip(),
        "chat_history":       st.session_state.chat_history[-6:],
        "emotion":            "neutral",
        "emotion_confidence": "low",
        "emotion_reasoning":  "",
        "retrieved_context":  "",
        "has_documents":      st.session_state.rag.has_documents,
        "response":           "",
        "hf_token":           st.session_state.hf_token,
        "model_name":         st.session_state.model_name,
    }

    with st.spinner("🤗 Thinking…"):
        try:
            result = graph.invoke(initial_state)
        except Exception as e:
            err = str(e)
            if "401" in err or "authorization" in err.lower():
                st.error("❌ Invalid HuggingFace token. Please check your token in the sidebar.")
            elif "loading" in err.lower() or "503" in err:
                st.warning("⏳ Model is loading on HuggingFace servers — please wait 30 seconds and try again.")
            else:
                st.error(f"❌ Error: {err}")
            st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_text.strip()})
    st.session_state.chat_history.append({
        "role":       "assistant",
        "content":    result["response"],
        "emotion":    result["emotion"],
        "confidence": result["emotion_confidence"],
        "reasoning":  result["emotion_reasoning"],
    })
    st.session_state.current_emotion = result["emotion"]

    if result.get("retrieved_context"):
        with st.expander("📄 Retrieved context used", expanded=False):
            st.markdown(f"""
            <div style='font-size:12px; color:#888; background:#12121c;
                        border-radius:10px; padding:14px; border:1px solid #2a2a3e;
                        white-space:pre-wrap; font-family:monospace;'>
{result["retrieved_context"][:1400]}{'…' if len(result["retrieved_context"])>1400 else ''}
            </div>""", unsafe_allow_html=True)

    st.rerun()
