#  Brain Tumor Classification using Deep Learning

This project presents an end-to-end deep learning pipeline for classifying brain tumors in MRI scans using both a **Custom Convolutional Neural Network (CNN)** and a **fine-tuned MobileNetV2** model. A Streamlit web application is also integrated to allow real-time image uploads and predictions.

---

## Key Features

-  **Two Deep Learning Approaches**: 
  - Custom-built CNN
  - Fine-tuned MobileNetV2 (Transfer Learning)
  
-  **Streamlit Web Application**: Upload and classify MRI brain images directly through a user-friendly interface.

-  **Extensive Evaluation**: Classification report, confusion matrix, and performance comparison between both models.

-  **Modular Codebase**: Separate scripts for preprocessing, training, inference, and deployment.

-  **Pretrained Models Included**: Ready-to-use `.h5` model files for instant predictions.

---

##  Dataset Overview

- MRI images categorized into:
  - `glioma`
  - `meningioma`
  - `pituitary`
  - `no_tumor`
  

- Images are resized to `224x224` and normalized to [0, 1] for model compatibility.

---

## 🔍 Model Architectures

### 1. Custom CNN

- 3 Conv2D + MaxPooling blocks
- Flatten → Dense (ReLU) → Dropout → Softmax
- Optimizer: Adam | Loss: Categorical Crossentropy

### 2. Fine-Tuned MobileNetV2

- Pretrained on ImageNet
- Unfrozen top 30 layers for fine-tuning
- Added Global Average Pooling, Dense, Dropout, and classification layers
- Achieved **80% accuracy** on the test set

---

## 📊 Results & Evaluation

| Metric        | Custom CNN | Fine-Tuned MobileNetV2 |
|---------------|------------|-------------------------|
| Accuracy       | 25%        | **80%**                 |
| Macro F1 Score | 0.24       | **0.78**                |
| Weighted F1    | 0.24       | **0.77**                |

- Detailed per-class precision, recall, and confusion matrix available in the report.
- `no_tumor` and `pituitary` classes were most accurately predicted by MobileNetV2.

---

## 🌐 Streamlit Web App

### Run Locally:
```bash
streamlit run app.py
```
```
├── app.py                                # Streamlit web application
├── classification.ipynb                 # Notebook for training/evaluation
├── fine_tuned_mobilenetv2_brain_tumor.h5  # Fine-tuned model file
├── brain_tumor_mobilenetv2_model.h5       # Earlier pretrained model
├── requirements.txt                     # Python dependencies
├── Brain Tumor Classification report.pdf # Final model report
├── LICENSE
└── README.md
```




