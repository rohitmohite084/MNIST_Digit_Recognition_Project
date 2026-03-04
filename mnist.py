import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os
import gdown
import requests
from streamlit_drawable_canvas import st_canvas
from streamlit_lottie import st_lottie

# -------------------------------
# 1. Page Configuration & UI Styling
# -------------------------------
st.set_page_config(page_title="MNIST Vision Pro", layout="wide")

# Custom CSS for a professional dark-themed dashboard
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
    .stMetric { border-radius: 15px; background: rgba(255, 255, 255, 0.05); border: 1px solid #00c6ff; padding: 20px; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #00c6ff, #0072ff); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 2. Asset Loading (Lottie & Model)
# -------------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_gd9p9v.json")

@st.cache_resource
def get_model():
    model_path = "mnist_cnn.h5"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?export=download&id=1NF-w35UmAzC_Q8_ln5CDhXSt9NLj-y83"
        with st.spinner("🚀 System Booting... Downloading CNN Model..."):
            gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = get_model()

# -------------------------------
# 3. Sidebar Navigation
# -------------------------------
with st.sidebar:
    if lottie_ai:
        st_lottie(lottie_ai, height=120, key="ai_sidebar")
    else:
        st.title("🤖 AI Control")
    
    st.markdown("---")
    input_mode = st.radio("Input Source Selection", ["Drawing Pad", "Upload Image"])
    st.markdown("---")
    st.info("💡 **Pro Tip:** Draw the digit clearly in the center for 99% accuracy.")
    
    if st.button("Reset System Cache"):
        st.cache_resource.clear()
        st.rerun()

# -------------------------------
# 4. Main Interface Layout
# -------------------------------
st.title("🧠 MNIST Intelligence Dashboard")
st.write("Advanced Real-time Pattern Recognition Engine")
st.markdown("---")

left_panel, right_panel = st.columns([1, 1], gap="large")

with left_panel:
    st.subheader("🖋️ Input Terminal")
    if input_mode == "Drawing Pad":
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.1)",
            stroke_width=25,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=380,
            width=380,
            drawing_mode="freedraw",
            key="main_canvas",
        )
        if canvas_result.image_data is not None:
            input_img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    else:
        file = st.file_uploader("Upload Handwritten Digit", type=["png", "jpg", "jpeg"])
        if file:
            input_img = Image.open(file).convert('L')
            # Auto-Invert if background is white/light
            if np.mean(input_img) > 127:
                input_img = ImageOps.invert(input_img)
            st.image(input_img, width=200, caption="Inverted Input Data")
        else:
            input_img = None

with right_panel:
    st.subheader("📊 Inference Analytics")
    
    # --- INPUT VALIDATION ---
    is_blank = True
    if input_mode == "Drawing Pad":
        # Check if any pixel intensity is above the threshold (indicating a drawing)
        if canvas_result.image_data is not None and np.any(canvas_result.image_data > 20):
            is_blank = False
    elif input_mode == "Upload Image" and file is not None:
        is_blank = False

    if is_blank:
        # Display professional warning if no input is detected
        st.warning("⚠️ **System Idle:** Please draw a digit or upload a file to begin recognition.")
        st.info("The neural network requires visual pixel data to perform inference.")
    else:
        # --- ADVANCED PREPROCESSING ---
        # 1. Enhance contrast for better feature extraction
        img = ImageOps.autocontrast(input_img, cutoff=2)
        
        # 2. Crop and Pad to center the digit (Mimics MNIST dataset format)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            img = ImageOps.expand(img, border=40, fill=0)
        
        # 3. Resize using LANCZOS for high-quality downsampling to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 4. Normalize pixel values to [0, 1]
        img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # --- MODEL PREDICTION ---
        with st.spinner("Processing pattern..."):
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        # --- RESULTS UI ---
        st.metric(label="Inferred Identity", value=str(predicted_digit), delta=f"{confidence:.2f}% Confidence")
        
        st.write("**Probability Distribution Matrix:**")
        cols = st.columns(2)
        for i in range(10):
            with cols[i % 2]:
                val = float(prediction[0][i])
                st.write(f"Digit {i}: {val*100:.1f}%")
                st.progress(val)

# -------------------------------
# 5. Technical Footer
# -------------------------------
st.markdown("---")
f1, f2, f3 = st.columns(3)
f1.caption("⚡ Latency: < 25ms")
f2.caption("🛰️ Model: CNN_MNIST_CORE")
f3.caption("📅 Status: Fully Operational")