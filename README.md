# 🎙️ NeuroPlacement AI

An end-to-end AI-powered placement prediction system built using a completely custom Neural Network implemented from scratch using NumPy.

NeuroPlacement AI combines:
- 🧠 Deep Learning Fundamentals
- 🎤 Speech-to-Text AI
- 🔊 Voice-Based Interaction
- 📊 Explainable AI Concepts
- 🌐 Real-Time Deployment

Unlike traditional ML projects that rely entirely on TensorFlow or Scikit-learn models, this project focuses heavily on understanding and implementing the internal mechanics of neural networks manually.

---

# 🚀 Live Demo

🔗 [Add Your Deployment Link Here]

---

# 📓 Kaggle / Notebook

🔗 [Add Your Kaggle Notebook Link Here]

---

# 🖼️ Project Preview

## Main Interface
![App UI](ADD_IMAGE_LINK_HERE)

---

## Voice Prediction Demo
![Voice Demo](ADD_IMAGE_LINK_HERE)

---

## Neural Network Prediction Output
![Prediction Output](ADD_IMAGE_LINK_HERE)

---

## Training Loss Visualization
![Loss Graph](ADD_IMAGE_LINK_HERE)

---

# 🧠 About The Project

NeuroPlacement AI predicts whether a student is likely to get placed based on:
- Academic Performance
- Technical Experience
- Communication Skills
- Placement Preparation
- Extracurricular Activities

The system supports:
- Manual Input
- Voice-Based Input
- Real-Time AI Predictions

The main goal of this project was not just achieving high accuracy, but deeply understanding:
- Forward Propagation
- Backpropagation
- Optimization
- Activation Functions
- Gradient Flow
- Real-world AI System Integration

---

# 🔥 Core Highlight — Custom Neural Network From Scratch

One of the biggest highlights of this project is that the neural network was implemented manually without using deep learning frameworks like TensorFlow or PyTorch.

The following components were built from scratch using NumPy:

✅ Forward Propagation  
✅ Backpropagation  
✅ Gradient Descent  
✅ Mini-Batch Training  
✅ Momentum Optimization  
✅ L2 Regularization  
✅ Dropout Regularization  
✅ He Initialization  
✅ ReLU / LeakyReLU Activations  
✅ Weight Updates  
✅ Loss Calculation  
✅ Multi-Layer Architecture  

This project was built mainly to understand how deep learning actually works internally rather than using black-box frameworks.

---

# 🧠 Neural Network Architecture

```text
Input Layer (10 Features)
        ↓
Hidden Layer (32 Neurons - LeakyReLU)
        ↓
Output Layer (1 Neuron - Sigmoid)
```

---

# 📊 Dataset Information

Dataset Source:
- Hugging Face Dataset Hub

Dataset Used:
- Campus Recruitment CSV Dataset

🔗 Dataset Link:
https://huggingface.co/datasets/Krooz/Campus_Recruitment_CSV

---

## Dataset Features

| Feature | Description |
|---|---|
| CGPA | Student CGPA |
| Internships | Internship Count |
| Projects | Number of Projects |
| Workshops/Certifications | Technical Exposure |
| AptitudeTestScore | Aptitude Score |
| SoftSkillsRating | Communication & Soft Skills |
| ExtracurricularActivities | Activities Participation |
| PlacementTraining | Placement Preparation |
| SSC_Marks | 10th Marks |
| HSC_Marks | 12th Marks |
| PlacementStatus | Target Variable |

---

## Dataset Statistics

- Total Samples: 7225
- Features: 10
- Binary Classification Task
- Balanced placement distribution

---

# ⚙️ Data Preprocessing Pipeline

The following preprocessing steps were performed:

✅ Null Value Checking  
✅ Label Encoding  
✅ One-Hot Encoding  
✅ Train-Test Split  
✅ Feature Scaling  
✅ Feature Transformation  
✅ Data Normalization  

The preprocessing pipeline was carefully designed to ensure stable neural network training and balanced gradient flow.

---

# 🎤 Voice AI Pipeline

The project supports real-time voice-based prediction.

Pipeline Flow:

```text
Voice Input
    ↓
Whisper Speech-to-Text
    ↓
Regex-Based Feature Extraction
    ↓
Feature Scaling
    ↓
Custom Neural Network
    ↓
Placement Prediction
```

---

# 🧠 Speech Recognition

Speech recognition was implemented using OpenAI Whisper.

Whisper converts user speech into structured text which is then processed using regex-based extraction to generate feature vectors for the neural network.

Example Voice Input:

```text
CGPA 8.2 internships 2 projects 3 aptitude 85 soft skills 4.5
```

---

# 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~79.6% |
| RMSE | ~0.37 |
| Baseline Improvement | ~43% |

The model demonstrated stable learning behavior and meaningful prediction trends across multiple stress-test scenarios.

---

# 🧪 Model Testing

The model was tested using:
- Strong Candidate Profiles
- Weak Candidate Profiles
- Extreme Inputs
- Mid-Level Profiles
- Skill-Dominant Profiles
- Academic-Dominant Profiles

This helped evaluate:
- Generalization
- Stability
- Feature Sensitivity
- Prediction Consistency

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| NumPy | Neural Network Computation |
| Pandas | Data Processing |
| Streamlit | Web Application |
| Whisper | Speech-to-Text |
| Scikit-learn | Preprocessing |
| Pickle | Model Saving |
| Matplotlib | Visualization |

---

# 📂 Project Structure

```bash
NeuroPlacement-AI/
│
├── model/
│   └── placement_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── dataset/
```

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuroPlacement-AI.git
cd NeuroPlacement-AI
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 🔮 Future Improvements

- Binary Cross Entropy Loss
- Better Explainable AI
- Live Microphone Streaming
- Advanced NLP Parsing
- Resume-Based Placement Prediction
- Multi-Class Package Prediction
- Real-Time Analytics Dashboard

---

# 📚 Major Learning Outcomes

This project helped build deep understanding of:

- Neural Network Internals
- Gradient Flow
- Optimization Techniques
- Regularization
- Model Deployment
- AI Product Design
- Voice-Based AI Systems
- End-to-End ML Pipelines

---

# 👨‍💻 Author

Aryan

AI/ML Enthusiast | Neural Networks | Deep Learning | AI Systems

---

# ⭐ Support

If you liked this project, consider giving it a ⭐ on GitHub.
