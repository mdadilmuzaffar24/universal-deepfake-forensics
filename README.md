# 🛡️ Universal Deepfake Forensics Engine

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An **Interpretable and Trustworthy Deepfake Detection Framework** engineered to authenticate high-fidelity media. This project moves beyond "black-box" classification by integrating mathematical transparency through Explainable AI (XAI), providing visual proof of algorithmic facial manipulation.

🔗 **[Live Application Server](https://universal-deepfake-forensics.streamlit.app/)**

---

## 🚀 Project Mission: Preserving Digital Integrity
With the rapid acceleration of Generative Adversarial Networks (GANs) and Autoencoders, synthetic media has reached hyper-realistic thresholds. This framework is designed to detect microscopic blending seams and textural anomalies left behind by facial manipulation algorithms, prioritizing model interpretability alongside raw predictive power.

---

## 🧠 Core Architecture Pipeline

The engine operates on a rigorously tested 3-phase neural network pipeline:

1. **Geometric Isolation (MTCNN):** A Multi-Task Cascaded Convolutional Neural Network detects and isolates the facial geometry, systematically cropping out irrelevant background noise to prevent the model from learning false environmental correlations.

2. **Textural Audit (Xception Backbone):** A heavily fine-tuned, transfer-learned Xception network analyzes the isolated facial topology. By leveraging depthwise separable convolutions, the model hunts for microscopic mathematical anomalies and facial blending seams unseen by the human eye.

3. **Explainable AI (XAI) Transparency:** To ensure the framework is trustworthy, decisions are visually mapped:
   * **Grad-CAM:** Provides spatial localization (heatmaps) highlighting the exact textural blending errors driving the network's verdict.
   * **SHAP:** Executes detailed attribute modeling, mapping pixel-level feature attribution to show which specific areas pushed the probability toward "Authentic" or "Fake."

---

## 📊 Model's Performance & Metrics

The framework was trained and evaluated on high-fidelity deepfake datasets, achieving exceptional class separability and robust predictive power.

* **Validation Accuracy:** 98.42%
* **Receiver Operating Characteristic (ROC-AUC):** 0.9989
* **False Positive Rate:** < 0.02

## 📈 Model's Performance Visualization

**Training vs. Validation Accuracy**

![Training vs Validation Accuracy](https://github.com/user-attachments/assets/d9776978-0eab-4666-8a66-a8d2119c91ac)

**Model Loss Progression**

![Model Loss Progression](https://github.com/user-attachments/assets/82959eab-e021-4182-9d0e-9055c3e9d2f6)

**Validation Confusion Matrix**

![Validation Confusion Matrix](https://github.com/user-attachments/assets/ebe253cd-1221-43ff-aad2-b4ec47cfe838)

**Receiver Operating Characteristic**

![ROC Curve](https://github.com/user-attachments/assets/ecb41a65-66ea-437e-a0f2-9c95642a95fe)

*(You can view the full Training Loss, Accuracy Progression, and Confusion Matrix within the 'Architecture & Metrics' tab of the live application also.)*

---

## 💻 Installation & Local Deployment

To run this forensic framework locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/mdadilmuzaffar24/universal-deepfake-forensics.git
cd universal-deepfake-forensics
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Engine
```bash
streamlit run app.py
```

> **Note:** The application is configured to automatically download the heavily optimized `.weights.h5` file from GitHub Releases upon first boot to bypass Git Large File Storage (LFS) constraints.

---

## 📁 Repository Structure

```plaintext
universal-deepfake-forensics/
├── app.py                  # Main Streamlit application and UI routing
├── requirements.txt        # Dependency tree (TensorFlow, MTCNN, SHAP, etc.)
├── .gitignore              # Ignored files (virtual environments, raw weights)
├── assets/                 # UI assets, model metrics, and benchmarks
│   ├── accuracy_graph.png
│   ├── loss_graph.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── models/                 # Directory auto-populated with weights on runtime
```
## 📥 Model Download

Due to GitHub's file size limits, the 238MB Rehearsal-Trained Xception model is hosted on Google Drive.

* [Download the .keras model here](https://drive.google.com/file/d/14UTCVCF6lVpYyx-5hnzBZ9sVImmKjWvx/view?usp=sharing)
* Place the downloaded file inside a `models/` directory in the root of this project before running `app.py`

---

## 🔮 Limitations & Future Scope

Currently, the Xception backbone acts as a highly specialized localized texture-hunter, excelling at finding the mathematical seams where a GAN pasted a fake face onto a real head.

**Future Iterations will target:**

- **Fully Synthetic Diffusion Models:** Integrating Frequency Domain Analysis (DCT/FFT) to catch uniform noise patterns generated by models like Midjourney or Stable Diffusion, where traditional blending seams do not exist.

- **Global Context:** Incorporating Vision Transformers (ViTs) to analyze asymmetric lighting and geometric anomalies across the entire image field.

---

## 👨‍💻 Architect & Developer

**MD ADIL MUZAFFAR** | Built as an M.Tech Data Science Capstone Project.

---
