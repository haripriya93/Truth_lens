#  TruthLens v3.2 — AI-Powered Image Authenticity Detector

**TruthLens** is an elegant and intelligent Streamlit dashboard that analyzes any image to detect whether it is **AI-generated or real**.  
It uses a blend of **heuristic image processing (OpenCV)** and **Gemini AI** scoring to produce a human-understandable authenticity score.

---

##  Features

- 🧠 **AI Integration (Gemini-Pro)** — Estimates AI-generation probability numerically  
- 🔍 **Heuristic Analysis** — Checks blur variance, entropy, symmetry, and EXIF metadata  
- 🖼️ **Real vs AI Verdict Tag** — Displayed on top of the uploaded image  
- 📊 **Beautiful Dashboard** — Includes metrics, charts, blur heatmaps, and brightness histograms  
- 🧾 **Human Summary Box** — Explains analysis in simple language  
- 📈 **Confidence Bar** — Visualizes AI vs Real confidence  
- 📷 **Image Facts** — Resolution, aspect ratio, file size  
- 💾 **Download JSON Report** — Exports results for sharing or record keeping  
- 🌙 **Dark UI** — Modern and professional Streamlit layout

---

##  Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit (Dark-themed dashboard) |
| **AI Model** | Google Gemini-Pro API |
| **Image Processing** | OpenCV, NumPy |
| **Data Visualization** | Matplotlib |
| **Language** | Python 3.11+ |


