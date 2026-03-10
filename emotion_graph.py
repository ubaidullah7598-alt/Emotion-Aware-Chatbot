"""
emotion_graph.py  –  HuggingFace Direct API (StopIteration fix)
────────────────────────────────────────────────────────────────
Calls huggingface_hub.InferenceClient directly — bypasses all
LangChain HF wrappers that trigger the StopIteration bug.

Pipeline: START → detect_emotion → retrieve_context → generate_response → END
"""

from __future__ import annotations
from typing import TypedDict, List, Dict
import json, re

from huggingface_hub import InferenceClient
from langgraph.graph import StateGraph, END


# ══════════════════════════════════════════════
#  Graph State
# ══════════════════════════════════════════════
class EmotionState(TypedDict):
    user_input:         str
    chat_history:       List[Dict[str, str]]
    emotion:            str
    emotion_confidence: str
    emotion_reasoning:  str
    retrieved_context:  str
    has_documents:      bool
    response:           str
    hf_token:           str
    model_name:         str


# ══════════════════════════════════════════════
#  Direct HF Inference helper
# ══════════════════════════════════════════════
def _hf_generate(state: EmotionState, prompt: str,
                 max_new_tokens: int = 400,
                 temperature: float = 0.7) -> str:
    """Call HuggingFace Inference API directly — no LangChain wrapper."""
    client = InferenceClient(
        model=state.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3"),
        token=state["hf_token"],
        timeout=60,
    )
    result = client.text_generation(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 0.01),
        do_sample=True,
        stop_sequences=["###", "</s>", "<|endoftext|>", "\nUser:"],
    )
    return result.strip() if isinstance(result, str) else str(result).strip()


# ══════════════════════════════════════════════
#  NODE 1 – Emotion Detection
# ══════════════════════════════════════════════
_DETECT_PROMPT = """[INST] You are an emotion analyst.
Read the message and respond with ONLY a JSON object. No explanation, no markdown.

Message: "{message}"

JSON format:
{{"emotion": "<happy|excited|neutral|confused|frustrated|sad|angry|anxious|stressed>", "confidence": "<low|medium|high>", "reasoning": "<one short sentence>"}}
[/INST]"""


def detect_emotion(state: EmotionState) -> EmotionState:
    try:
        raw = _hf_generate(state, _DETECT_PROMPT.format(message=state["user_input"]),
                           max_new_tokens=100, temperature=0.01)
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        parsed = json.loads(match.group() if match else raw)
        emotion    = parsed.get("emotion",    "neutral")
        confidence = parsed.get("confidence", "medium")
        reasoning  = parsed.get("reasoning",  "")
    except Exception:
        raw_lower = (raw if 'raw' in dir() else "").lower()
        emotion = "neutral"
        for emo in ["happy","excited","confused","frustrated","sad","angry","anxious","stressed"]:
            if emo in raw_lower:
                emotion = emo
                break
        confidence = "low"
        reasoning  = "Detected from keywords."

    valid = {"happy","excited","neutral","confused","frustrated","sad","angry","anxious","stressed"}
    emotion = emotion if emotion in valid else "neutral"

    return {**state, "emotion": emotion,
            "emotion_confidence": confidence,
            "emotion_reasoning": reasoning}


# ══════════════════════════════════════════════
#  NODE 3 – Emotion-Aware Response Generation
# ══════════════════════════════════════════════
EMOTION_STYLE: Dict[str, str] = {
    "happy":      "Be warm, upbeat, and celebratory.",
    "excited":    "Be enthusiastic and match their energy.",
    "neutral":    "Be clear, balanced, and informative.",
    "confused":   "Be patient. Explain simply, step by step.",
    "frustrated": "Acknowledge their frustration calmly, then help solve it.",
    "sad":        "Be gentle, compassionate, and supportive.",
    "angry":      "Stay calm. Validate their feelings without escalating.",
    "anxious":    "Be reassuring and grounding. Give structured guidance.",
    "stressed":   "Be practical. Break things into small clear steps.",
}

_RESPONSE_PROMPT = """[INST] You are an empathetic AI assistant.
The user's emotion is: {emotion}. Tone: {style}

Rules:
1. First sentence must warmly acknowledge the user's emotion.
2. Then answer their question helpfully and concisely.
3. Sound human and caring — never robotic.
{rag_block}{history_block}
User message: {user_input}
[/INST]"""

def generate_response(state: EmotionState) -> EmotionState:
    emotion = state.get("emotion", "neutral")
    style   = EMOTION_STYLE.get(emotion, "Be helpful and supportive.")
    context = state.get("retrieved_context", "")

    rag_block = ""
    if context:
        rag_block = f"\nRelevant document context (cite the source if used):\n---\n{context[:800]}\n---\n"

    history_lines = []
    for m in state.get("chat_history", [])[-4:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {m['content']}")
    history_block = ("\nPrevious conversation:\n" + "\n".join(history_lines) + "\n") \
                    if history_lines else ""

    prompt = _RESPONSE_PROMPT.format(
        emotion=emotion, style=style,
        rag_block=rag_block, history_block=history_block,
        user_input=state["user_input"],
    )

    try:
        response = _hf_generate(state, prompt, max_new_tokens=400, temperature=0.7)
        # Clean up any prompt leakage
        for marker in ["[/INST]", "[INST]", "User message:", "User:"]:
            if marker in response:
                response = response.split(marker)[-1].strip()
    except Exception as e:
        response = f"I'm having trouble responding right now. Error: {e}"

    return {**state, "response": response}


# ══════════════════════════════════════════════
#  Graph Builder
# ══════════════════════════════════════════════
def build_graph(rag_system=None):
    def retrieve_context(state: EmotionState) -> EmotionState:
        context = ""
        if state.get("has_documents") and rag_system is not None:
            context = rag_system.retrieve(state["user_input"])
        return {**state, "retrieved_context": context}

    wf = StateGraph(EmotionState)
    wf.add_node("detect_emotion",    detect_emotion)
    wf.add_node("retrieve_context",  retrieve_context)
    wf.add_node("generate_response", generate_response)
    wf.set_entry_point("detect_emotion")
    wf.add_edge("detect_emotion",    "retrieve_context")
    wf.add_edge("retrieve_context",  "generate_response")
    wf.add_edge("generate_response", END)
    return wf.compile()
