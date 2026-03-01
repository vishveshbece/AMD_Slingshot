"""
app.py — Kria AI Classroom Assistant · Streamlit GUI
=====================================================
Displays:
  • Real-time camera feed (via OpenCV → st.image)
  • Lab experiment step tracker (from LabStateMachine)
  • DPU inference results for the current frame
  • AI Concept Coach chat interface (local LLM via OpenAI-compatible API)

Run:
    streamlit run app.py

Environment variables (optional):
    CAMERA_INDEX        — OpenCV camera index (default: 0)
    XMODEL_PATH         — Path to compiled .xmodel (default: models/demo.xmodel)
    LLM_BASE_URL        — Base URL for local LLM (default: http://localhost:11434/v1)
    LLM_MODEL           — Model name (default: llama3)
"""

import os
import sys
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import requests

# Allow importing sibling packages when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from state_machine.state_machine import LabStateMachine
from inference.inference_runner   import DPUInferenceRunner

log = logging.getLogger("ClassroomApp")

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
XMODEL_PATH  = os.getenv("XMODEL_PATH",  "models/demo.xmodel")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "llama3")
LABELS_PATH  = os.getenv("LABELS_PATH",  "models/labels.txt")

REFRESH_INTERVAL_MS = 100   # camera frame refresh rate

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kria AI Classroom Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state initialisation (runs once per browser session)
# ---------------------------------------------------------------------------
def _init_session():
    if "sm" not in st.session_state:
        def hint_callback(msg: str):
            st.session_state.setdefault("hints", []).append(msg)

        st.session_state.sm = LabStateMachine(hint_callback=hint_callback)

    if "runner" not in st.session_state:
        st.session_state.runner = DPUInferenceRunner(
            XMODEL_PATH,
            use_hw_preprocess=True,
            labels_path=LABELS_PATH if Path(LABELS_PATH).exists() else None,
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role":    "assistant",
                "content": (
                    "👋 Hi! I'm your AI Concept Coach. "
                    "Ask me anything about FPGAs, neural networks, or today's lab!"
                ),
            }
        ]

    if "hints" not in st.session_state:
        st.session_state.hints = []

    if "last_top5" not in st.session_state:
        st.session_state.last_top5 = []

_init_session()

sm:     LabStateMachine   = st.session_state.sm
runner: DPUInferenceRunner = st.session_state.runner

# ---------------------------------------------------------------------------
# Helper: capture one frame from the camera
# ---------------------------------------------------------------------------
@st.cache_resource
def get_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

camera = get_camera(CAMERA_INDEX)

def read_frame() -> np.ndarray | None:
    ok, frame = camera.read()
    if not ok:
        return None
    return frame   # BGR uint8

# ---------------------------------------------------------------------------
# Helper: call the local LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert AI tutor embedded in a Kria FPGA classroom
assistant. You explain complex topics (deep learning, FPGA architecture, Vitis AI,
heterogeneous computing) clearly and concisely for university students.
Keep answers under 150 words unless asked for more detail."""

def ask_llm(user_message: str, history: list[dict]) -> str:
    """
    Sends chat history + new message to a local OpenAI-compatible endpoint
    (e.g., Ollama, LM Studio, llama.cpp server).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json={"model": LLM_MODEL, "messages": messages, "max_tokens": 200},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Local LLM not reachable. "
            f"Make sure Ollama (or compatible server) is running at `{LLM_BASE_URL}`."
        )
    except Exception as exc:
        return f"⚠️ LLM error: {exc}"

# ===========================================================================
# ─── LAYOUT ────────────────────────────────────────────────────────────────
# ===========================================================================

st.title("🎓 Kria AI Classroom Assistant")
st.caption("Heterogeneous CPU+FPGA edge AI — AMD Kria KV260")

# ─── Sidebar: Lab Experiment Controls ───────────────────────────────────────
with st.sidebar:
    st.header("🔬 Lab Experiment")
    st.metric("Current Step", sm.current_step)
    st.progress(sm.progress_pct / 100)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Start Setup",      use_container_width=True):
            sm.advance("start_setup")
        if st.button("⚙ Begin Processing", use_container_width=True):
            sm.advance("begin_processing")
    with col_b:
        if st.button("✅ Verify",           use_container_width=True):
            sm.advance("verify")
        if st.button("🔄 Reset",            use_container_width=True):
            sm.advance("reset")

    # Skip button (demonstrates hint system)
    if st.button("⏭ Skip to Verify (test skip hint)", use_container_width=True):
        sm.advance("skip_to_verify")

    # Display any pending hints
    if st.session_state.hints:
        st.divider()
        st.subheader("💡 Hints")
        for hint in st.session_state.hints:
            st.warning(hint)
        if st.button("Clear Hints"):
            st.session_state.hints.clear()
            st.rerun()

# ─── Main area: two columns ─────────────────────────────────────────────────
cam_col, chat_col = st.columns([3, 2], gap="large")

# ── Camera Feed + Inference Results ─────────────────────────────────────────
with cam_col:
    st.subheader("📷 Live Camera Feed")
    frame_placeholder = st.empty()
    inference_placeholder = st.empty()

    frame = read_frame()
    if frame is not None:
        # Run inference only in Processing / Verification states
        if sm.state in ("processing", "verification"):
            try:
                probs = runner.infer(frame)
                st.session_state.last_top5 = runner.top_k(probs, k=5)
            except Exception as exc:
                log.error("Inference error: %s", exc)

        # Overlay top prediction on frame
        if st.session_state.last_top5:
            top = st.session_state.last_top5[0]
            label_text = f"{top['label']}  {top['confidence']*100:.1f}%"
            cv2.putText(
                frame, label_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 80), 2, cv2.LINE_AA
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
    else:
        frame_placeholder.warning("⚠️ Camera not detected. Check CAMERA_INDEX.")

    # Inference results table
    if st.session_state.last_top5:
        with inference_placeholder.container():
            st.caption("🔍 Top-5 DPU Predictions")
            for pred in st.session_state.last_top5:
                conf = pred["confidence"]
                st.write(f"**{pred['label']}**")
                st.progress(conf, text=f"{conf*100:.2f}%")

    # Auto-refresh (re-runs the script on a timer)
    time.sleep(REFRESH_INTERVAL_MS / 1000)
    st.rerun()

# ── AI Concept Coach Chat ────────────────────────────────────────────────────
with chat_col:
    st.subheader("🤖 AI Concept Coach")

    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask about FPGAs, neural networks, or the lab…")
    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.spinner("Thinking…"):
            reply = ask_llm(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply}
        )
        st.rerun()