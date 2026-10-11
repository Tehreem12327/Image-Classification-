import numpy as np
import cv2
import streamlit as st
from tensorflow import keras  # if this errors, use: import keras

st.set_page_config(page_title="PetVision AI", page_icon="🐾", layout="centered")

CATEGORIES = ["Cat", "Dog"]
EMOJI = {"Cat": "🐱", "Dog": "🐶"}

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(circle at 20% 10%, #1e1b4b 0%, #0f172a 45%, #020617 100%);
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 720px; padding-top: 3rem; }

.hero { text-align: center; margin-bottom: 2rem; }
.hero h1 {
    font-size: 3rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; font-size: 1.05rem; margin-top: .5rem; }

[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(129,140,248,0.5);
    border-radius: 20px; padding: 2rem;
    transition: all .2s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #c084fc; background: rgba(255,255,255,0.07);
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px; padding: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    margin-top: 1.5rem; text-align: center;
}
.badge {
    display: inline-block; padding: .5rem 1.4rem; border-radius: 999px;
    font-size: 1.6rem; font-weight: 800; color: white;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    margin-top: 1rem;
}
.label { color: #94a3b8; font-size: .85rem; letter-spacing: .1em; text-transform: uppercase; }
.bar-wrap { background: rgba(255,255,255,0.1); border-radius: 999px; height: 12px; margin-top: .6rem; overflow: hidden; }
.bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #22d3ee, #818cf8, #f472b6); }
.conf { font-size: 1.1rem; font-weight: 600; margin-top: .6rem; color: #e2e8f0; }
.split { display: flex; gap: 1rem; margin-top: 1.2rem; }
.split div { flex: 1; background: rgba(255,255,255,0.05); border-radius: 14px; padding: .8rem; }
.split b { font-size: 1.2rem; }
.foot { text-align: center; color: #64748b; font-size: .8rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_resource
def load_model():
    return keras.models.load_model("model.h5", compile=False)

model = load_model()

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>PetVision AI</h1>
  <p>Upload a photo and our model will tell you whether it's a cat or a dog.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")

# ---------- Prediction ----------
if uploaded is not None:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img_color = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img_color is None:
        st.error("Could not read this image. Please try another file.")
    else:
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        arr = cv2.resize(gray, (60, 60)).reshape(-1, 60, 60, 1)

        with st.spinner("Analyzing image..."):
            pred = model.predict(arr, verbose=0)[0]

        idx = int(np.argmax(pred))
        label = CATEGORIES[idx]
        conf = float(pred[idx]) * 100

        left, mid, right = st.columns([1, 3, 1])
        with mid:
            st.image(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Raw outputs may not sum to 1 if the model has no softmax; show them as-is
        scores = "".join(
            f"<div><span class='label'>{c}</span><br><b>{float(pred[i])*100:.1f}%</b></div>"
            for i, c in enumerate(CATEGORIES)
        )

        st.markdown(f"""
        <div class="card">
          <span class="label">Prediction</span><br>
          <span class="badge">{EMOJI[label]} {label}</span>
          <div class="conf">Confidence: {conf:.1f}%</div>
          <div class="bar-wrap"><div class="bar" style="width:{min(conf,100):.1f}%"></div></div>
          <div class="split">{scores}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="foot">Built with Streamlit &amp; Keras · CNN trained on cats vs dogs</div>',
            unsafe_allow_html=True)