import os
import urllib.request
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import shap
import matplotlib.colors as mcolors
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception

# --- PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="Deepfake Forensics Engine", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    [data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #050A18 !important;
        background-image: radial-gradient(circle at 50% 0%, #112240 0%, #050A18 70%) !important;
        color: #E0E0E0 !important;
    }
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
        border-color: rgba(0, 250, 154, 0.5) !important; 
    }
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
    .status-warning { color: #FFB84C !important; font-weight: 700 !important; font-size: 22px !important; letter-spacing: 0.5px !important; }
    .status-danger { color: #FF4B4B !important; font-weight: 700 !important; font-size: 22px !important; letter-spacing: 0.5px !important; }
    .status-success { color: #00FA9A !important; font-weight: 700 !important; font-size: 22px !important; letter-spacing: 0.5px !important; }
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
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

# --- SIDEBAR: HOW IT WORKS ---
with st.sidebar:
    st.header("⚙️ Engine Architecture")
    st.markdown("This framework utilizes a 3-step media forensics pipeline:")
    st.markdown("1. **Geometric Isolation:** MTCNN detects and isolates the facial geometry, discarding irrelevant background data.")
    st.markdown("2. **Textural Audit:** A transfer-learned Xception network analyzes the face for microscopic blending seams and GAN artifacts.")
    st.markdown("3. **Explainable AI (XAI):** Grad-CAM and SHAP visualize the exact regions and pixels driving the network's verdict.")
    st.markdown("---")
    st.info("Capstone Project Framework")

# --- INITIALIZE SESSION STATE ---
if 'shap_executed' not in st.session_state:
    st.session_state.shap_executed = False
if 'shap_plot' not in st.session_state:
    st.session_state.shap_plot = None
if 'processed_image_key' not in st.session_state:
    st.session_state.processed_image_key = None

# --- 1. DEFINE EXACT NESTED SKELETON ---
def build_forensics_model(input_shape=(299, 299, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # 1. Internal Augmentation
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
    ], name="Augmentation_Layer")
    x = augmentation(inputs)
    
    # 2. Internal Rescaling
    x = layers.Rescaling(1./255, name="rescaling")(x)
    
    # 3. Nested Xception Backbone
    xception_base = Xception(weights=None, include_top=False, input_shape=input_shape)
    xception_base._name = 'xception' 
    x = xception_base(x)
    
    # 4. Top Layers
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    predictions = layers.Dense(1, activation='sigmoid', name="Forensic_Verdict")(x)
    
    model = models.Model(inputs=inputs, outputs=predictions)
    return model

# --- 2. DEFINE LOADER ---
@st.cache_resource
def load_forensics_engine_v3():
    model_dir = './models'
    weights_path = f'{model_dir}/xception_weights_only.weights.h5'
    
    if not os.path.exists(weights_path):
        st.info("☁️ Cloud Server Initializing: Downloading weights from GitHub Releases...")
        os.makedirs(model_dir, exist_ok=True)
        url = 'https://github.com/mdadilmuzaffar24/universal-deepfake-forensics/releases/download/v2.0/xception_weights_only.weights.h5'
        urllib.request.urlretrieve(url, weights_path)
        st.success("✅ Model Download Complete!")

    model = build_forensics_model()
    # Strict loading to ensure perfect matching
    model.load_weights(weights_path)
    model.trainable = False
    return model

@st.cache_resource
def load_face_detector():
    return MTCNN()

# --- 3. EXECUTE LOADERS ---
try:
    model = load_forensics_engine_v3()
    detector = load_face_detector()
    st.sidebar.success("✅ Universal Framework Loaded")
    st.sidebar.success("✅ MTCNN Geometry Engine Loaded")
except Exception as e:
    st.sidebar.error(f"Engine Load Error: {e}")

# --- GRAD-CAM ENGINE ---
def generate_gradcam(img_array, model, pred_value):
    target_tensor = np.expand_dims(img_array, axis=0)
    
    xception_base = model.get_layer('xception')
    last_conv_layer_name = "block13_sepconv2_act"
    
    grad_model = tf.keras.models.Model(
        [xception_base.inputs], 
        [xception_base.get_layer(last_conv_layer_name).output, xception_base.output]
    )
    
    with tf.GradientTape() as tape:
        # Preprocessing is handled cleanly by the model's own layers
        preprocessed = model.get_layer('Augmentation_Layer')(target_tensor, training=False)
        preprocessed = model.get_layer('rescaling')(preprocessed)
        
        last_conv_output, xception_output = grad_model(preprocessed)
        
        preds = model.get_layer('global_average_pooling2d')(xception_output)
        preds = model.get_layer('dropout')(preds, training=False)
        preds = model.get_layer('Forensic_Verdict')(preds)
        
        if pred_value < 0.5:
            class_channel = 1.0 - preds[:, 0]
        else:
            class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_output[0]
    heatmap_raw = tf.squeeze(last_conv_layer_output @ pooled_grads[..., tf.newaxis]).numpy()

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
    def predict_wrapper(x_batch):
        # Model handles rescaling internally now!
        return model.predict(x_batch, verbose=0)
    
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
    ax.imshow(sv_spatial, cmap=cmap, vmin=-vmax, vmax=vmax, alpha=0.6, interpolation='bicubic')
    ax.axis('off')
    
    return fig

# --- UI WORKFLOW ---
# --- UI WORKFLOW (TABBED INTERFACE) ---
# Create two professional tabs
tab1, tab2 = st.tabs(["🛡️ Forensics Scanner", "📊 Architecture & Metrics"])

# ==========================================
# TAB 1: THE MAIN SCANNER
# ==========================================
with tab1:
    # Update the uploader
    uploaded_file = st.file_uploader(
        "Drop a target image here...", 
        type=["jpg", "png", "jpeg"], 
        help="Upload a clear, front-facing image. The engine requires a visible human face to perform the textural audit."
    )

    st.caption("ℹ️ **Engine Scope:** This framework is trained on high-fidelity deepfake datasets to detect facial manipulation, blending boundaries, and face-swaps. It is not designed to detect fully synthetic AI-generated art (e.g., Midjourney) where no facial blending occurred.")

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
            
            if not faces:
                st.error("🚨 Architectural Rejection: No human face detected.")
                st.warning("This engine requires a visible human face to perform a deepfake textural audit. Please upload an image containing a clear facial subject.")
                st.stop()
                
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            cropped_face = img_array_raw[y:y+h, x:x+w]

            cropped_pil = Image.fromarray(cropped_face)
            img_resized = cropped_pil.resize((299, 299), resample=Image.NEAREST)
            img_array = np.array(img_resized)  
            
            target_tensor = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(target_tensor, verbose=0)[0][0]
            overlay_image = generate_gradcam(img_array, model, prediction)

        st.subheader("Architectural Verdict")
        if prediction > 0.70:
            st.markdown('<p class="status-success">✅ AUTHENTIC MEDIA (HIGH CONFIDENCE)</p>', unsafe_allow_html=True)
            # Add tooltip here
            st.metric(label="Authenticity Score", value=f"{prediction * 100:.2f}%", help="100% indicates zero detected manipulation. Scores above 70% are considered highly authentic.")
            verdict_text = "The network found no traces of algorithmic facial manipulation or blending seams."
        elif prediction < 0.30:
            st.markdown('<p class="status-danger">🚨 DEEPFAKE DETECTED (HIGH CONFIDENCE)</p>', unsafe_allow_html=True)
            # Add tooltip here
            st.metric(label="Manipulation Score", value=f"{(1.0 - prediction) * 100:.2f}%", help="Higher scores indicate a higher probability of mathematical anomalies and blending seams.")
        else:
            st.markdown('<p class="status-warning">⚠️ SUSPICIOUS MEDIA (HUMAN REVIEW REQUIRED)</p>', unsafe_allow_html=True)
            conf = max(prediction, 1.0 - prediction) * 100
            st.metric(label="Inconclusive Confidence", value=f"{conf:.2f}%")
            verdict_text = "Proceed with caution. The network cannot definitively verify facial authenticity."

        st.write(verdict_text)
        # Update the XAI subheader
        st.markdown(
            "### Explainable AI (XAI) Evidence", 
            help="XAI provides transparent, visual proof of how the neural network arrived at its decision, ensuring the results are trustworthy rather than a 'black box'."
        )
        
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
                        shap_figure = generate_shap_plot(img_array, model, max_evals=1000)
                        st.session_state.shap_plot = shap_figure
                        st.session_state.shap_executed = True
                        st.rerun()

            if st.session_state.shap_executed:
                st.pyplot(st.session_state.shap_plot, use_container_width=True)
                st.markdown("*Red pixels push toward Fake, Green pixels push toward Authentic.*")

# ==========================================
# TAB 2: ARCHITECTURE & METRICS
# ==========================================
with tab2:
    st.header("Interpretable and Trustworthy Deepfake Detection Framework")
    st.markdown("**Leveraging Transfer-Learned CNNs with Grad-CAM and SHAP for Robust Media Forensics**")
    
    st.markdown("---")
    
    st.subheader("Core Architecture")
    st.markdown("""
    This engine is built on a dual-phase pipeline designed for both accuracy and transparency:
    1. **Geometric Isolation:** An MTCNN (Multi-Task Cascaded Convolutional Neural Network) detects and extracts facial geometry to remove background noise.
    2. **Feature Extraction:** A heavily fine-tuned **Xception** (and parallel ResNet50 experimental) architecture serves as the backbone. The model utilizes transfer learning to identify microscopic blending artifacts left behind by GANs and autoencoders.
    3. **Explainable AI (XAI):** Decisions are mapped visually using Grad-CAM (Gradient-weighted Class Activation Mapping) for spatial localization and SHAP (SHapley Additive exPlanations) for pixel-level feature attribution.
    """)
    
    st.markdown("---")
    
    st.subheader("Training Performance & Metrics")
    
    # We create two columns to display your Colab graphs side-by-side
    # --- ROW 1: Accuracy & Loss ---
    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        st.markdown("##### Training vs. Validation Accuracy")
        st.image("assets/accuracy_graph.png", use_container_width=True)
    with graph_col2:
        st.markdown("##### Model Loss Progression")
        st.image("assets/loss_graph.png", use_container_width=True)
        
    st.markdown("---")
    
    # --- ROW 2: Confusion Matrix & ROC ---
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.markdown("##### Validation Confusion Matrix")
        st.image("assets/confusion_matrix.png", use_container_width=True)
        st.caption("Demonstrates high True Positive/Negative rates with minimal misclassification.")
    with metric_col2:
        st.markdown("##### Receiver Operating Characteristic")
        st.image("assets/roc_curve.png", use_container_width=True)
        st.caption("An exceptional AUC of 0.9989, proving robust class separability.")
        
    st.markdown("---")
    st.markdown("""
    *The framework ensures high-fidelity media authentication by prioritizing model interpretability alongside raw predictive power.*
    """)

# --- GLOBAL FOOTER ---
st.markdown("""
    <div class="custom-footer">
        Architected and Developed by <span>MD ADIL MUZAFFAR</span>
    </div>
""", unsafe_allow_html=True)