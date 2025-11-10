import streamlit as st
import numpy as np
import subprocess
import sys
import cv2
from PIL import Image, ExifTags
import os
import io
import json
import matplotlib.pyplot as plt
import google.generativeai as genai

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="TruthLens v3.2 — AI Image Detector", layout="wide", page_icon="🤖")

st.title("🤖 TruthLens  — AI-Powered Image Authenticity Detector")
st.caption("Analyze and visualize the authenticity of any image using heuristic + Gemini AI scoring.")

# ---------- GEMINI CONFIG ----------
genai.configure(api_key="AIzaSyYOUR_REAL_GEMINI_KEY_HERE")  # Replace with your Gemini key

# ---------- HELPER FUNCTIONS ----------
def pil_to_cv(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def variance_of_laplacian(gray): return cv2.Laplacian(gray, cv2.CV_64F).var()

def left_right_symmetry(gray):
    h, w = gray.shape
    mid = w // 2
    left, right = gray[:, :mid], gray[:, w - mid:]
    right_flipped = cv2.flip(right, 1)
    return np.mean(np.abs(left.astype(np.float32) - right_flipped.astype(np.float32))) / 255.0

def image_entropy(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_norm = hist / hist.sum()
    return -np.sum(hist_norm * np.log2(hist_norm + 1e-7))

def extract_exif(image_pil):
    try:
        exif = image_pil._getexif()
        if exif:
            return True, {ExifTags.TAGS.get(k,k):v for k,v in exif.items()}
        return False, {}
    except: return False, {}

def ai_assess(signals):
    prompt = f"""
    Return a number between 0 and 1 showing chance of being AI-generated.
    EXIF: {signals['exif_present']}
    Blur: {signals['laplacian_variance']}
    Symmetry: {signals['symmetry_mad']}
    Entropy: {signals['entropy_bits']}
    """
    try:
        r = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        val = float(r.text.strip())
        return val if 0<=val<=1 else 0.5
    except: return 0.5

# ---------- SIDEBAR ----------
st.sidebar.header("🧭 Navigation")
up = st.sidebar.file_uploader("📤 Upload an Image", type=["jpg","jpeg","png"])
run = st.sidebar.button("🚀 Analyze")

# ---------- MAIN ----------
if run and up:
    img = Image.open(up)
    st.image(img, use_container_width=True)

    # Basic image facts
    img_cv = pil_to_cv(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    w, h = img.size
    file_size = len(up.getvalue()) / 1024  # KB

    lap = variance_of_laplacian(gray)
    ent = image_entropy(gray)
    sym = left_right_symmetry(gray)
    exif_ok, exif = extract_exif(img)

    # Heuristic scoring
    risk = 0
    if not exif_ok: risk += .25
    if lap < 100:   risk += .25
    if sym < .05:   risk += .25
    if ent < 4:     risk += .25
    heur = min(1.0, risk)

    ai_risk = ai_assess({
        "exif_present": exif_ok, "laplacian_variance": lap,
        "symmetry_mad": sym, "entropy_bits": ent
    })
    final = round(heur*0.6 + ai_risk*0.4, 2)

    # ---------- OVERLAY VERDICT ----------
    verdict_text = "LIKELY REAL" if final < 0.4 else "POSSIBLY AI" if final < 0.7 else "LIKELY AI-GENERATED"
    color = (0,255,0) if final < 0.4 else (0,165,255) if final < 0.7 else (0,0,255)
    overlay_img = img_cv.copy()
    cv2.putText(overlay_img, verdict_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption=f"Verdict: {verdict_text}", use_container_width=True)

    # ---------- METRICS ----------
    st.markdown("## 📊 Detailed Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Blur Variance", f"{lap:.1f}")
    c2.metric("Entropy", f"{ent:.2f}")
    c3.metric("Symmetry MAD", f"{sym:.3f}")
    c4.metric("AI Likelihood", f"{ai_risk*100:.1f}%")

    # ---------- AI Confidence Bar ----------
    st.markdown("### 🧠 AI Confidence Comparison")
    st.write(f"🧍 Real: **{(1-final)*100:.1f}%** | 🤖 AI: **{final*100:.1f}%**")
    bar = int(final * 50)
    st.text("[" + "█"*bar + "░"*(50-bar) + "]")

    # ---------- HUMAN SUMMARY ----------
    with st.container():
        st.markdown(
            f"""
            <div style='background-color:#1C1F26;padding:15px;border-radius:12px'>
            <h4>🧠 Human Summary</h4>
            <p>
            The image appears to have a <b>{(1-final)*100:.1f}% chance of being natural</b>
            and a <b>{final*100:.1f}% chance of being AI-generated</b>.<br>
            Blur variance of <b>{lap:.1f}</b> and entropy of <b>{ent:.2f}</b>
            suggest {"good texture detail" if ent>5 else "smooth low-detail surfaces"}.
            {("No EXIF data was found, a common sign of generated content."
              if not exif_ok else "EXIF metadata is present, typical of real camera images.")}<br>
            Overall verdict: <b>{verdict_text}</b>.
            </p></div>
            """, unsafe_allow_html=True
        )

    # ---------- EXTRA VISUALS ----------
    st.markdown("### 📈 Histogram & Brightness Map")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(gray.ravel(), bins=256, color='cyan'); ax[0].set_title("Pixel Intensity Histogram")
    ax[1].imshow(gray, cmap='gray'); ax[1].set_title("Grayscale Map"); ax[1].axis('off')
    st.pyplot(fig)

    st.markdown("### 🔥 Blur Heatmap")
    lapmap = cv2.normalize(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    st.image(lapmap, caption="Blur Intensity Heatmap", use_container_width=True)

    # ---------- IMAGE FACTS ----------
    st.markdown("### 📷 Image Facts")
    st.write(f"- **Resolution:** {w} × {h} pixels")
    st.write(f"- **Aspect Ratio:** {w/h:.2f}")
    st.write(f"- **File Size:** {file_size:.1f} KB")
    st.write(f"- **Color Depth:** 3 channels (RGB)")

    # ---------- DOWNLOAD RESULTS ----------
    st.markdown("### 📥 Download Your Analysis")
# Convert all NumPy types to native Python types before JSON export
    result = {
        "verdict": str(verdict_text),
        "ai_probability": float(final),
        "blur_variance": float(lap),
        "entropy": float(ent),
        "symmetry_mad": float(sym),
        "exif_present": bool(exif_ok),
        "file_size_kb": float(round(file_size, 1)),
        "resolution": f"{w}x{h}"
}

    json_bytes = json.dumps(result, indent=4).encode("utf-8")

    st.download_button("⬇️ Download JSON Report", data=json_bytes, file_name="TruthLens_Report.json", mime="application/json")

    # ---------- FOOTER ----------
    st.markdown("---")
    st.caption("✨ TruthLens v3.2 — Smart. Simple. Stunning. Created by Hari priya ❤️")

else:
    st.info("Upload an image and click **Analyze** to start.")

  
