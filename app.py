import streamlit as st
import numpy as np
import cv2
from PIL import Image, ExifTags
import io

st.set_page_config(page_title="TruthLens — MVP", layout="wide")
st.title("🔍 TruthLens — MVP (Heuristic Authenticity Analyzer)")
st.caption("Upload an image and optionally paste a related claim/caption. This MVP uses simple, explainable checks (no heavy AI yet).")

# ---------- Helpers ----------
def pil_to_cv(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def left_right_symmetry(gray):
    h, w = gray.shape
    mid = w // 2
    left = gray[:, :mid]
    right = gray[:, w - mid:]
    right_flipped = cv2.flip(right, 1)
    # mean absolute difference normalized
    mad = np.mean(np.abs(left.astype(np.float32) - right_flipped.astype(np.float32))) / 255.0
    # smaller MAD => more symmetry
    return mad

def image_entropy(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    p = hist / (np.sum(hist) + 1e-8)
    p = p[p>0]
    ent = -np.sum(p * np.log2(p))
    return float(ent)  # 0..8 approx

def get_exif_presence(pil_img):
    try:
        exif = pil_img.getexif()
        if exif is None or len(exif) == 0:
            return False, 0
        return True, len(exif)
    except Exception:
        return False, 0

def clamp01(x): return max(0.0, min(1.0, float(x)))

def analyze_image(pil_img):
    cv_img = pil_to_cv(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # EXIF
    exif_present, exif_count = get_exif_presence(pil_img)
    exif_risk = 0.7 if not exif_present else 0.0  # missing EXIF increases risk

    # Blur/smoothness
    lap_var = variance_of_laplacian(gray)
    # If variance below ~150, likely smooth; normalize to risk 0..1
    blur_risk = clamp01((150.0 - min(lap_var, 150.0)) / 150.0)

    # Symmetry (lower MAD => more symmetry => higher risk)
    mad = left_right_symmetry(gray)
    symmetry_risk = clamp01((0.15 - min(mad, 0.15)) / 0.15)

    # Entropy (lower entropy => higher risk)
    ent = image_entropy(gray)  # roughly 0..8
    entropy_risk = clamp01((4.5 - min(ent, 4.5)) / 4.5)

    # Combine with weights
    weights = {
        "exif": 0.15,
        "blur": 0.30,
        "symmetry": 0.25,
        "entropy": 0.30
    }
    final_risk = (
        weights["exif"] * exif_risk +
        weights["blur"] * blur_risk +
        weights["symmetry"] * symmetry_risk +
        weights["entropy"] * entropy_risk
    )

    # Build explanations
    reasons = []
    if not exif_present:
        reasons.append("No camera EXIF metadata detected (common in generated/edited images, but also in social media re-uploads).")
    if blur_risk > 0.5:
        reasons.append(f"Very smooth/low-detail regions (Laplacian variance ~ {lap_var:.1f}).")
    if symmetry_risk > 0.5:
        reasons.append(f"High left–right symmetry (MAD ~ {mad:.3f}).")
    if entropy_risk > 0.5:
        reasons.append(f"Low texture diversity/entropy (~ {ent:.2f} bits).")

    return {
        "final_risk": clamp01(final_risk),
        "signals": {
            "exif_present": exif_present,
            "exif_count": int(exif_count),
            "laplacian_variance": float(lap_var),
            "symmetry_mad": float(mad),
            "entropy_bits": float(ent)
        },
        "reasons": reasons
    }

def analyze_claim(text):
    if not text or text.strip() == "":
        return {"risk": 0.0, "reasons": []}
    t = text.strip()
    risk = 0.0
    reasons = []

    # Heuristics
    if sum(1 for c in t if c.isupper()) > 0.5 * len(t.replace(" ", "")):
        risk += 0.2; reasons.append("Excessive UPPERCASE usage (sensational style).")
    if t.count("!") >= 3:
        risk += 0.2; reasons.append("Multiple exclamation marks (sensational style).")
    clickbait = ["shocking", "you won't believe", "secret revealed", "breaking", "viral", "exposed"]
    if any(kw in t.lower() for kw in clickbait):
        risk += 0.2; reasons.append("Clickbait phrasing detected.")
    hedges = ["experts say", "studies show", "researchers claim", "sources say"]
    if any(kw in t.lower() for kw in hedges):
        risk += 0.1; reasons.append("Vague sourcing/hedging language.")
    if len(t) > 240:
        risk += 0.1; reasons.append("Very long claim (often opinion or storytelling rather than factual).")

    return {"risk": clamp01(risk), "reasons": reasons}

# ---------- UI ----------
left, right = st.columns([1.3, 1])
with left:
    img_file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    claim = st.text_area("Optional: paste a related claim/caption to analyze writing style", height=120, placeholder="e.g., BREAKING!!! Aliens spotted above Mumbai...")
with right:
    st.markdown("**How it works (MVP):**")
    st.write("- Checks **EXIF, blur, symmetry, entropy** of the image")
    st.write("- Optionally checks claim **tone/style** for sensational markers")
    st.write("- Combines signals into a **risk score** (0 = authentic, 1 = likely synthetic/manipulated)")

go = st.button("Analyze")

if go:
    if img_file is None and (not claim or claim.strip()==""):
        st.warning("Please upload an image or paste a claim.")
        st.stop()

    img = None
    if img_file is not None:
        img = Image.open(io.BytesIO(img_file.read())).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

    img_result = {"final_risk": 0.0, "signals": {}, "reasons": []}
    if img is not None:
        img_result = analyze_image(img)

    claim_result = analyze_claim(claim)

    # Combine overall
    overall_risk = 0.0
    if img is not None and claim.strip() != "":
        overall_risk = clamp01(0.8 * img_result["final_risk"] + 0.2 * claim_result["risk"])
    elif img is not None:
        overall_risk = img_result["final_risk"]
    else:
        overall_risk = claim_result["risk"]

    verdict = "Likely Authentic" if overall_risk < 0.5 else "Likely AI-Generated / Manipulated"
    st.subheader("Result")
    st.metric("Verdict", verdict, delta=f"Risk {overall_risk*100:.1f}%")

    st.markdown("### Signals & Scores")
    colA, colB, colC = st.columns(3)
    with colA:
        st.write("**Image signals**")
        if img is None:
            st.write("_No image uploaded._")
        else:
            st.json(img_result["signals"])
    with colB:
        st.write("**Image explanations**")
        if img is None or not img_result["reasons"]:
            st.write("_No strong red flags detected by heuristics._")
        else:
            for r in img_result["reasons"]:
                st.write("• " + r)
    with colC:
        st.write("**Claim analysis**")
        if not claim or claim.strip()=="":
            st.write("_No claim provided._")
        else:
            st.write(f"Risk from writing style: **{claim_result['risk']*100:.0f}%**")
            for r in claim_result["reasons"]:
                st.write("• " + r)

    st.markdown("---")
    st.caption("This MVP uses simple heuristics. Real photos can trigger false positives (e.g., social media strips EXIF). Use these signals as guidance, not absolute proof.")
