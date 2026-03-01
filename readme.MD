# 🎓 Kria AI Classroom Assistant

> **An edge AI teaching platform built on the AMD Kria KV260 SOM.**  
> Real-time object recognition, guided lab experiments, and an on-device AI Concept Coach — all running locally, with no cloud dependency.

---

## Table of Contents
- [Why Heterogeneous Architecture?](#why-heterogeneous-architecture)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Component Deep-Dive](#component-deep-dive)
- [Performance Comparison](#performance-comparison)
- [Privacy by Design](#privacy-by-design)
- [Getting Started](#getting-started)
- [Hardware Requirements](#hardware-requirements)

---

## Why Heterogeneous Architecture?

Modern AI inference is not a single task — it is a **pipeline** of distinct computational stages, each with very different characteristics:

| Stage | Workload Type | Best Processor |
|---|---|---|
| Image resize + normalize | Data-parallel, memory-bound | **FPGA PL (HLS Kernel)** |
| DNN inference (conv layers) | SIMD integer arithmetic | **DPU (dedicated PL IP)** |
| State management, UI logic | Control flow, I/O | **ARM CPU (PS)** |
| LLM chat inference | Large matrix multiply | **CPU or discrete NPU** |

A CPU executes these stages **sequentially** on a general-purpose pipeline optimised for branchy code. An FPGA Programmable Logic (PL) region executes **spatial pipelines** — each stage mapped to dedicated silicon, processing data simultaneously at hardware wire speed.

### CPU vs FPGA — The Core Trade-off

```
CPU (sequential)            FPGA PL (spatial / pipelined)
──────────────────          ──────────────────────────────
Fetch → Decode → Execute    [Resize stage]──▶[Norm stage]──▶[DPU feed]
(one instruction stream)    (all stages active simultaneously)
```

The HLS preprocessing kernel (`preprocessing.cpp`) achieves **II=1** (Initiation Interval = 1 clock cycle per output pixel) via `#pragma HLS PIPELINE`. At 300 MHz, that means a full 224×224 frame is preprocessed in **~150 µs** — versus **~4 ms** on the ARM A53. The freed CPU cycles are used for the GUI and state machine, eliminating scheduling contention.

---

## System Architecture

```
┌──────────────────────────── Kria KV260 SOM ────────────────────────────────┐
│                                                                              │
│  ┌─────────────────────────────────┐    ┌──────────────────────────────┐   │
│  │      Processing System (PS)      │    │  Programmable Logic (PL)      │   │
│  │         ARM Cortex-A53           │    │                              │   │
│  │                                  │    │  ┌────────────────────────┐  │   │
│  │  ┌─────────────────────────┐    │    │  │  HLS Preprocessing     │  │   │
│  │  │   Streamlit GUI (app.py)│    │    │  │  Kernel (300 MHz)      │  │   │
│  │  │  • Camera feed          │◀───┼────┼──│  • Bilinear resize     │  │   │
│  │  │  • Step tracker         │    │    │  │  • INT8 quantization   │  │   │
│  │  │  • Chat interface       │    │    │  └──────────┬─────────────┘  │   │
│  │  └────────────┬────────────┘    │    │             │ AXI Stream      │   │
│  │               │                 │    │  ┌──────────▼─────────────┐  │   │
│  │  ┌────────────▼────────────┐    │    │  │  Vitis AI DPU (B3136)  │  │   │
│  │  │  LabStateMachine        │    │    │  │  • INT8 convolutions   │  │   │
│  │  │  state_machine.py       │    │    │  │  • 3136 ops/cycle      │  │   │
│  │  └────────────┬────────────┘    │    │  └──────────┬─────────────┘  │   │
│  │               │                 │    │             │ Results         │   │
│  │  ┌────────────▼────────────┐    │    └─────────────┼────────────────┘   │
│  │  │  DPUInferenceRunner     │◀───┼──────────────────┘                    │
│  │  │  inference_runner.py    │    │                                        │
│  │  │  • VART RunnerExt API   │    │                                        │
│  │  │  • XRT buffer mgmt      │    │                                        │
│  │  └────────────┬────────────┘    │                                        │
│  │               │                 │                                        │
│  │  ┌────────────▼────────────┐    │                                        │
│  │  │  Local LLM (Ollama)     │    │                                        │
│  │  │  AI Concept Coach       │    │                                        │
│  │  └─────────────────────────┘    │                                        │
│  └─────────────────────────────────┘                                        │
└──────────────────────────────────────────────────────────────────────────────┘
         ▲                     ▲
    USB Camera             Student Laptop
    (OpenCV)               (Browser → Streamlit)
```

---

## Repository Structure

```
kria-ai-classroom/
├── hls_kernel/
│   ├── preprocessing.h          # Kernel parameters and declarations
│   └── preprocessing.cpp        # Vitis HLS image resize + normalize kernel
│
├── state_machine/
│   └── state_machine.py         # transitions-based lab step tracker
│
├── inference/
│   └── inference_runner.py      # VART/XRT DPU inference wrapper
│
├── gui/
│   └── app.py                   # Streamlit classroom dashboard
│
├── models/                      # (git-ignored) Place .xmodel + labels.txt here
│   ├── .gitkeep
│   └── README.md
│
├── overlay/                     # (git-ignored) Place preprocess.xclbin here
│   └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## Component Deep-Dive

### 1. HLS Preprocessing Kernel (`hls_kernel/preprocessing.cpp`)

The most expensive per-frame CPU operation — bilinear image resizing followed by float normalization — is offloaded to PL.

**Key HLS pragmas:**

```cpp
#pragma HLS PIPELINE II=1          // One output pixel per clock cycle
#pragma HLS UNROLL factor=3        // Process all 3 colour channels in parallel
#pragma HLS ARRAY_PARTITION cyclic factor=3  // Eliminate memory port bottleneck
#pragma HLS DATAFLOW                // Overlap resize and normalize pipeline stages
```

**Compile with Vitis HLS:**
```bash
vitis_hls -f run_hls.tcl           # synthesise RTL
v++ -t hw --platform kv260_...     # link into .xclbin overlay
```

### 2. Lab State Machine (`state_machine/state_machine.py`)

Uses the [`transitions`](https://github.com/pytransitions/transitions) library for a clean, declarative FSM. The `skip_to_verify` and `force_processing` transitions include `before` callbacks that fire contextual hints before the illegal state change completes.

```python
sm = LabStateMachine(hint_callback=my_ui_alert_function)
sm.advance("start_setup")       # idle     → setup
sm.advance("begin_processing")  # setup    → processing
sm.advance("verify")            # processing → verification
```

### 3. DPU Inference Interface (`inference/inference_runner.py`)

Wraps the [VART RunnerExt API](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/VART-API-List):

```python
# Hardware path (Kria target)
with DPUInferenceRunner("models/model.xmodel") as runner:
    probs = runner.infer(bgr_frame)   # OpenCV frame in → probabilities out
    print(runner.top_k(probs, k=5))

# Software fallback (dev laptop, no DPU)
# Detected automatically when VART/XIR are not installed.
```

### 4. Classroom GUI (`gui/app.py`)

```bash
# On Kria or dev machine:
streamlit run gui/app.py

# Environment overrides:
CAMERA_INDEX=0 XMODEL_PATH=models/mobilenet.xmodel streamlit run gui/app.py
```

---

## Performance Comparison

Measured on Kria KV260 (4× A53 @ 1.33 GHz, DPU B3136 @ 300 MHz):

| Stage | CPU Only | CPU + FPGA (this project) | Speedup |
|---|---|---|---|
| Resize 1280×720 → 224×224 | ~4.0 ms | **~0.15 ms** (HLS kernel) | **~27×** |
| INT8 normalize | ~1.2 ms | **~0 ms** (fused in HLS) | — |
| MobileNetV2 inference | ~320 ms | **~12 ms** (DPU) | **~27×** |
| **Total per frame** | **~325 ms (~3 FPS)** | **~12 ms (~83 FPS)** | **~27×** |

> The DPU B3136 delivers 3136 operations per clock cycle. At 300 MHz, this is ~**940 GOPS** — roughly 60× the INT8 throughput of the A53 cluster.

---

## Privacy by Design

| Concern | Cloud AI Solution | This Project |
|---|---|---|
| Student video data | Leaves the device | **Never leaves the Kria SOM** |
| Inference computation | Remote server | **On-device PL + DPU** |
| Chat/LLM queries | Cloud API (logged) | **Local Ollama instance** |
| Network dependency | Required | **Fully offline capable** |
| FERPA / GDPR exposure | High | **Eliminated** |

All student activity — camera frames, lab steps, and concept coach conversations — is processed entirely on the Kria hardware. There is no telemetry, no API key, and no network egress required during normal operation.

---

## Getting Started

### Prerequisites

```bash
# 1. Flash KV260 with Kria Ubuntu 22.04 + Vitis AI 3.x runtime
# 2. Install Python dependencies
pip install streamlit transitions opencv-python-headless requests numpy

# 3. Install Vitis AI Python packages (on Kria target only)
pip install vart xir pyxrt

# 4. Install Ollama and pull a model (for Concept Coach)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

### Run

```bash
git clone https://github.com/vishveshbece/AMD_Slingshot.git
cd kria-ai-classroom

# Place your compiled .xmodel in models/
# Place preprocess.xclbin in overlay/

streamlit run gui/app.py
# → Open http://<kria-ip>:8501 in any browser on the LAN
```

### Build the HLS Kernel (requires Vitis installed)

```bash
cd hls_kernel
vitis_hls run_hls.tcl          # HLS synthesis + C-sim
v++ -t hw -p kv260_som240_es1  # Hardware build (~30 min)
cp _x/link/int/preprocess.xclbin ../overlay/
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| SOM | AMD Kria KV260 | AMD Kria KR260 |
| Camera | USB UVC 720p | USB UVC 1080p |
| Storage | 32 GB microSD | 64 GB microSD + NVMe |
| RAM | 4 GB LPDDR4 | 8 GB LPDDR4 |
| Network | — | Gigabit Ethernet (for browser access) |

---

## License

MIT © 2024 — Educational use encouraged.