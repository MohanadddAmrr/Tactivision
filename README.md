# TactiVision Pro

> End-to-end computer vision pipeline that converts broadcast football video into 70+ tactical features. Built as a graduation project with a 6-person team.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Demo

**Live Streamlit dashboard:** [coming soon]
**Technical writeup:** [coming soon]
**Demo video:** [coming soon]

![TactiVision demo](docs/demo.gif)

---

## What it does

TactiVision Pro ingests raw broadcast football video and outputs 70+ tactical features used by analysts and coaches:

- Player and ball detection and tracking
- Pass detection, pass networks, and pass completion rates
- Shot detection with expected goals (xG) modeling
- Formation detection across 17 template formations
- Pressing analysis (PPDA)
- Player identity resolution via jersey OCR + roster matching
- Heat maps, possession stats, pitch control estimates

A 16-tab Streamlit dashboard presents everything interactively.

---

## Key results

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| Ball detection (per-frame) | 4.7% | **85.7%** | +81.0 pp |
| ID fragmentation vs BoT-SORT | — | **-13.75%** | — |
| Pipeline throughput | — | **25 FPS** | — |
| Peak RAM | — | 249 MB | — |
| Tracked frames (3 full matches) | — | ~120,000 | — |

The ball detection jump was the hardest technical problem in the pipeline. Broadcast ball tracking fails on motion blur, occlusion, and small apparent size on wide shots. We solved it by stacking two detectors: a fine-tuned YOLOv8n for the common case, with an RT-DETR second-pass dedicated to the frames YOLO missed.

---

## Architecture

```
Broadcast video
      │
      ▼
┌──────────────────────┐
│  Frame extraction    │  (OpenCV)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  YOLOv8n detection   │  (players, ball — primary)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  RT-DETR second pass │  (ball recovery on failed frames)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ByteTrack           │  (identity continuity across frames)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Feature extractors  │  (70+ tactical features)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Streamlit dashboard │  (16 tabs, interactive)
└──────────────────────┘
```

---

## Tech stack

- **Language:** Python 3.10+
- **Deep learning:** PyTorch, Ultralytics (YOLOv8), RT-DETR
- **Tracking:** ByteTrack
- **Computer vision:** OpenCV
- **OCR:** EasyOCR (jersey number recognition)
- **Dashboard:** Streamlit
- **Data:** SQLite (roster matching), pandas, NumPy

---

## Repository structure

```
TactiVision/
├── src/
│   ├── detection/         # YOLOv8 + RT-DETR dual-model system
│   ├── tracking/          # ByteTrack integration
│   ├── features/          # 70+ tactical feature extractors
│   ├── identity/          # OCR + roster matching
│   └── dashboard/         # Streamlit app
├── models/                # Trained weights (download separately)
├── notebooks/             # Research and benchmark notebooks
├── docs/                  # Diagrams, demo gifs, writeups
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone
git clone https://github.com/MohanadddAmrr/TactiVision.git
cd TactiVision

# Create virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Download pretrained weights (link in /models/README.md)

# Run the dashboard
streamlit run src/dashboard/app.py
```

---

## The team

Built with a six-person team as the graduation project for the BSc Data Science program at Misr International University (MIU), 2026.

My role: Team lead and ML/CV engineer.

---

## Writeup

For the full technical story behind the 4.7% → 85.7% ball detection jump, read the Medium post: [link coming soon]

---

## License

MIT. See [LICENSE](LICENSE).
