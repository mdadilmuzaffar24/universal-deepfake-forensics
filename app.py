import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import shap
import matplotlib.colors as mcolors
import urllib.request


# --- PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="Deepfake Forensics Engine", page_icon="🛡️", layout="wide")

# --- AGGRESSIVE SAAS CSS INJECTION ---
st.markdown("""
    <style>
    /* 1. Force Hide Streamlit Header & Footer */
    [data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    .stDeployButton { display: none !important; }

    /* 2. Global App Background (Deep Cyber Navy Gradient) */
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #050A18 !important;
        background-image: radial-gradient(circle at 50% 0%, #112240 0%, #050A18 70%) !important;
        color: #E0E0E0 !important;
    }

    /* 3. Modern Glassmorphism Metric Cards */
    [data-testid="stMetric"] {
        background-color: rgba(17, 34, 64, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        text-align: center !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(0, 250, 154, 0.5) !important; /* Subtle green glow on hover */
    }

    /* 4. Style the Drag-and-Drop Uploader */
    [data-testid="stFileUploader"] > div > div {
        background-color: rgba(17, 34, 64, 0.4) !important;
        border: 1px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"] > div > div:hover {
        border-color: #00FA9A !important;
        background-color: rgba(17, 34, 64, 0.7) !important;
    }

    /* 5. Sleek Call-to-Action Buttons */
    .stButton>button {
        width: 100% !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%) !important;
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.6) !important;
        transform: translateY(-2px) !important;
        border-color: rgba(255,255,255,0.3) !important;
    }

    /* 6. Typography and Status Colors */
    .status-warning { color: #FFB84C !important; font-weight: 700 !important; font-size: 22px !important; letter-spacing: 0.5px !important; }
    .status-danger { color: #FF4B4B !important; font-weight: 700 !important; font-size: 22px !important; letter-spacing: 0.5px !important; }
    .status-success { color: #00FA9A !important; font-weight: 700 !important; font-size: 22px !important; letter-spacing: 0.5px !important; }
    
    /* 7. Image Container Borders */
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* 8. Custom Branding Footer */
    .custom-footer {
        text-align: center;
        padding: 30px 0px 10px 0px;
        margin-top: 60px;
        color: #8b9bb4;
        font-size: 14px;
        letter-spacing: 1px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    .custom-footer span {
        color: #00FA9A;
        font-weight: 600;
        text-shadow: 0px 0px 10px rgba(0, 250, 154, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Universal Deepfake Forensics Engine")
st.markdown("Upload a high-fidelity image. The engine will isolate facial geometry and perform a textural audit.")

# --- INITIALIZE SESSION STATE FOR XAI WORKFLOW ---
if 'shap_executed' not in st.session_state:
    st.session_state.shap_executed = False
if 'shap_plot' not in st.session_state:
    st.session_state.shap_plot = None
if 'processed_image_key' not in st.session_state:
    st.session_state.processed_image_key = None


import os
import urllib.request
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from mtcnn import MTCNN

# --- 1. REBUILD THE SKELETON (This is what got deleted!) ---
def build_forensics_model(input_shape=(299, 299, 3)):
    base_model = Xception(weights=None, include_top=False, input_shape=input_shape)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x) 
    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

# --- 2. INJECT THE BRAIN (CACHE-BUSTED) ---
@st.cache_resource
def load_forensics_engine_v2():
    model_dir = './models'
    weights_path = f'{model_dir}/xception_weights_only.weights.h5'
    
    if not os.path.exists(weights_path):
        st.info("☁️ Cloud Server Initializing: Downloading weights from GitHub Releases...")
        os.makedirs(model_dir, exist_ok=True)
        
        url = 'https://github.com/mdadilmuzaffar24/universal-deepfake-forensics/releases/download/v2.0/xception_weights_only.weights.h5'
        urllib.request.urlretrieve(url, weights_path)
        st.success("✅ Model Download Complete!")

    # Now it knows exactly what build_forensics_model is!
    model = build_forensics_model()
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    model.trainable = False
    
    return model

@st.cache_resource
def load_face_detector():
    return MTCNN()

try:
    model = load_forensics_engine_v2()
    detector = load_face_detector()
    st.sidebar.success("✅ Universal Framework Loaded")
    st.sidebar.success("✅ MTCNN Geometry Engine Loaded")
except Exception as e:
    st.sidebar.error(f"Engine Load Error: {e}")

# ... (Keep your generate_gradcam and everything else below here) ...

# --- GRAD-CAM ENGINE (REFINED SHARPNESS) ---
def generate_gradcam(img_array, model, pred_value):
    # 1. Rescale the input exactly how ImageDataGenerator did
    x = np.expand_dims(img_array, axis=0) / 255.0
    
    # 2. Xception layers are now directly accessible
    last_conv_layer_name = "block13_sepconv2_act"
    
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(x)
        
        if pred_value < 0.5:
            class_channel = 1.0 - preds[:, 0]
        else:
            class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_output[0]
    heatmap_raw = tf.squeeze(last_conv_layer_output @ pooled_grads[..., tf.newaxis]).numpy()

    # REFINED NORM & CONTRAST
    heatmap = np.maximum(heatmap_raw, 0)
    heatmap = np.sqrt(heatmap) 

    if np.max(heatmap) == 0:
        heatmap = np.abs(heatmap_raw)

    heatmap /= np.max(heatmap) + 1e-8
    heatmap_resized = cv2.resize(heatmap, (299, 299))
    heatmap_smoothed = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
    heatmap_color = plt.get_cmap('jet')(heatmap_smoothed)[:, :, :3]
    
    img_array_normalized = img_array / 255.0
    overlay = heatmap_color * 0.5 + img_array_normalized * 0.5
    return np.clip(overlay, 0, 1)

# --- SHAP EXPLAINER ENGINE ---
def generate_shap_plot(img_array, model, max_evals=1000):
    # CRITICAL: Scale the SHAP batches to 0.0 - 1.0
    def predict_wrapper(x_batch):
        return model.predict(x_batch / 255.0, verbose=0)
    
    target_tensor = np.expand_dims(img_array, axis=0)
    
    masker = shap.maskers.Image("blur(10,10)", target_tensor[0].shape)
    explainer = shap.Explainer(predict_wrapper, masker, output_names=["Authenticity"])
    
    shap_values = explainer(target_tensor, max_evals=max_evals, batch_size=50)
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_backdrop = cv2.GaussianBlur(img_gray, (3, 3), 0)
    ax.imshow(img_backdrop, cmap='gray', alpha=1.0) 
    
    sv = shap_values.values[0]
    if len(sv.shape) == 4: 
        sv = sv[..., 0]
    sv_spatial = np.sum(sv, axis=-1)
    
    cmap = mcolors.LinearSegmentedColormap.from_list("RedGreen", ["#FF0000", "#FFFFFF", "#00FF00"])
    
    vmax = np.max(np.abs(sv_spatial))
    im = ax.imshow(sv_spatial, cmap=cmap, vmin=-vmax, vmax=vmax, alpha=0.6, interpolation='bicubic')
    ax.axis('off')
    
    return fig

# --- UI WORKFLOW ---
uploaded_file = st.file_uploader("Drop a target image here...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    current_image_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.processed_image_key != current_image_key:
        st.session_state.shap_executed = False
        st.session_state.shap_plot = None
        st.session_state.processed_image_key = current_image_key

    st.markdown("---")
    pil_image = Image.open(uploaded_file).convert('RGB')
    
    with st.spinner("Executing textural audit (automatic mode)..."):
        img_array_raw = np.array(pil_image)
        faces = detector.detect_faces(img_array_raw)
        
        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            cropped_face = img_array_raw[y:y+h, x:x+w]
        else:
            cropped_face = img_array_raw
            st.sidebar.warning("⚠️ MTCNN failed to detect face. Analyzing full frame.")

        cropped_pil = Image.fromarray(cropped_face)
        img_resized = cropped_pil.resize((299, 299), resample=Image.NEAREST)
        img_array = np.array(img_resized)  
        
        # 1. Prepare the tensor
        target_tensor = np.expand_dims(img_array, axis=0)
        
        # 2. CRITICAL: Rescale the tensor for the prediction
        target_tensor_scaled = target_tensor / 255.0
        
        # 3. Feed the scaled tensor to the model
        prediction = model.predict(target_tensor_scaled, verbose=0)[0][0]
        
        # 4. Generate Grad-CAM (passing the unscaled img_array is fine, it scales internally)
        overlay_image = generate_gradcam(img_array, model, prediction)

    st.subheader("Architectural Verdict")
    if prediction > 0.70:
        st.markdown('<p class="status-success">✅ AUTHENTIC MEDIA (HIGH CONFIDENCE)</p>', unsafe_allow_html=True)
        st.metric(label="Authenticity Score", value=f"{prediction * 100:.2f}%")
        verdict_text = "The network found no traces of algorithmic manipulation."
    elif prediction < 0.30:
        st.markdown('<p class="status-danger">🚨 DEEPFAKE DETECTED (HIGH CONFIDENCE)</p>', unsafe_allow_html=True)
        st.metric(label="Manipulation Score", value=f"{(1.0 - prediction) * 100:.2f}%")
        verdict_text = "The network detected mathematical anomalies and blending seams."
    else:
        st.markdown('<p class="status-warning">⚠️ SUSPICIOUS MEDIA (HUMAN REVIEW REQUIRED)</p>', unsafe_allow_html=True)
        conf = max(prediction, 1.0 - prediction) * 100
        st.metric(label="Inconclusive Confidence", value=f"{conf:.2f}%")
        verdict_text = "Proceed with caution. The network cannot definitively verify authenticity."

    st.write(verdict_text)
    st.markdown("---")
    
    st.subheader("Explainable AI (XAI) Evidence")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("1. MTCNN Isolated Target")
        st.image(cropped_pil, use_container_width=True)
        
    with col2:
        st.caption("2. Grad-CAM (Heatmap)")
        st.image(overlay_image, use_container_width=True, clamp=True)
        st.markdown("*Highlights localized textural blending errors.*")
        
    with col3:
        st.caption("3. SHAP (Pixel Attribution)")
        
        if not st.session_state.shap_executed:
            st.info("SHAP analysis is computationally intensive. Click below to execute detailed attribute modeling (~45s).")
            if st.button("Execute Detailed Attribute Analysis"):
                with st.spinner("Executing 1000 SHAP evaluations..."):
                    # Updated to 1000 to match the new optimized function
                    shap_figure = generate_shap_plot(img_array, model, max_evals=1000)
                    st.session_state.shap_plot = shap_figure
                    st.session_state.shap_executed = True
                    st.rerun()

        if st.session_state.shap_executed:
            st.pyplot(st.session_state.shap_plot, use_container_width=True)
            st.markdown("*Red pixels push toward Fake, Green pixels push toward Authentic.*")

# --- CUSTOM BRANDING FOOTER (MOVED TO GLOBAL SCOPE) ---
st.markdown("""
    <div class="custom-footer">
        Architected and Developed by <span>MD ADIL MUZAFFAR</span>
    </div>
""", unsafe_allow_html=True)


