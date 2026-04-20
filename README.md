# 🛡️ Universal Deepfake Forensics Engine
**Architected by MD Adil Muzaffar**

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

An enterprise-grade, end-to-end deep learning framework designed to detect AI-generated media (Deepfakes) using Transfer Learning and Explainable AI (XAI). This project addresses the critical need for "trust and transparency" in AI diagnostics by opening the black box of neural network decision-making.

### 🚀 Key Features
* **Multi-Domain Robustness:** The core model (Xception) was optimized using Rehearsal Training across three distinct datasets (FaceForensics++, Celeb-DF, and DFDC) to cure catastrophic forgetting.
* **Automated Geometry Extraction:** Utilizes MTCNN to automatically detect, isolate, and crop facial data from raw images.
* **Tri-State Confidence Logic:** Implements real-world risk management thresholds (Authentic, Suspicious, Deepfake) rather than simple binary classification.
* **Dual-XAI Pipeline:** * **Grad-CAM:** Semantic heatmaps highlighting localized textural blending errors.
  * **Targeted DeepSHAP:** Instantaneous neural backpropagation (Gradient x Input) isolating the top 30% of mathematical anomalies pushing the network's verdict.

### ⚙️ Local Installation
1. Clone the repository: `git clone https://github.com/mdadilmuzaffar24/universal-deepfake-forensics.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment and install dependencies: `pip install -r requirements.txt`
4. Launch the dashboard: `streamlit run app.py`

### 📥 Model Download
Due to GitHub's file size limits, the 238MB Rehearsal-Trained Xception model is hosted on Google Drive. 
* [Download the .keras model here](https://drive.google.com/file/d/14UTCVCF6lVpYyx-5hnzBZ9sVImmKjWvx/view?usp=sharing)
* Place the downloaded file inside a `models/` directory in the root of this project before running `app.py`.