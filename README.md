# 🧠 EmoSense RAG — HuggingFace Edition (100% Free)

> LangGraph · LangChain · FAISS · Streamlit · HuggingFace Inference API

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Get your **free** HuggingFace token at https://huggingface.co/settings/tokens  
→ New token → Role: **Read** → Copy & paste into the sidebar.

---

## 🤗 Available Models (all free)

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | ⚡ Fast | ⭐⭐⭐⭐ | **Recommended** |
| `HuggingFaceH4/zephyr-7b-beta` | ⚡ Fast | ⭐⭐⭐⭐ | Great for chat |
| `meta-llama/Meta-Llama-3-8B-Instruct` | ⚡ Fast | ⭐⭐⭐⭐⭐ | Requires HF access |
| `google/gemma-2-2b-it` | ⚡⚡ Very fast | ⭐⭐⭐ | Lightweight |
| `microsoft/Phi-3-mini-4k-instruct` | ⚡⚡ Very fast | ⭐⭐⭐ | Tiny & quick |

---

## 🔀 LangGraph Pipeline

```
START
  │
  ▼
detect_emotion        HF model classifies emotion as JSON
  │
  ▼
retrieve_context      FAISS top-4 similarity search
  │                   (auto-skipped if no documents loaded)
  ▼
generate_response     HF model crafts emotion-aware + RAG reply
  │
  ▼
END
```

---

## 🎭 Emotion → Response Style

| Emotion | Response Tone |
|---|---|
| 😊 Happy | Warm, upbeat, celebratory |
| 🤩 Excited | Enthusiastic, energetic |
| 😐 Neutral | Clear, precise |
| 😕 Confused | Patient, step-by-step |
| 😤 Frustrated | Calm, empathetic, solution-focused |
| 😢 Sad | Gentle, compassionate |
| 😠 Angry | Calm, validating |
| 😰 Anxious | Reassuring, structured |
| 😫 Stressed | Practical, concrete |

---

## 📚 RAG — Zero Cost

- Upload **PDFs, TXT, or Markdown** files in the sidebar
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings (no API key needed)
- FAISS in-memory index — fast similarity search
- Multi-document support
