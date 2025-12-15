# Deep-Learning-Model-For-Diabetes-Detection-through-Retinopathy

# Diabetic Retinopathy Classification (CNN)

## 📌 Project Overview

This project implements a **custom Convolutional Neural Network (CNN)** to classify **Diabetic Retinopathy (DR)** from retinal fundus images into **5 classes (0–4)**.
It was developed for the **GDG on Campus PIEAS AI/ML Hackathon**.

⚠️ **Important:** This project uses a **from-scratch CNN (no transfer learning)**, and results reflect that limitation.

---

## 🧠 DR Classes

| Label | Description      |
| ----- | ---------------- |
| 0     | No DR            |
| 1     | Mild             |
| 2     | Moderate         |
| 3     | Severe           |
| 4     | Proliferative DR |

---

## 📂 Dataset

* **Source:** Kaggle – *Diabetic Retinopathy Balanced Dataset*
* **Classes:** 5 (balanced)
* **Training images:** 27,834
* **Validation images:** 6,958
* **Image size:** 128 × 128

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Google Colab
* Kaggle API

---

## 🏗️ Model Architecture

Custom CNN implemented exactly as in code:

```
Input (128×128×3)
↓
Conv2D (32) + ReLU
MaxPooling2D
↓
Conv2D (64) + ReLU
MaxPooling2D
↓
Conv2D (128) + ReLU
MaxPooling2D
↓
Flatten
Dense (128) + ReLU
Dropout (0.4)
Dense (5) + Softmax
```

* **Total Parameters:** ~3.3M
* **Optimizer:** Adam (lr = 1e-4)
* **Loss:** Categorical Crossentropy
* **Metric:** Accuracy

---

## 🚀 Training Setup

* **Batch size:** 64
* **Epochs:** 20 (EarlyStopping enabled)
* **Data Augmentation:**

  * Rescaling
  * Rotation
  * Zoom
  * Horizontal flip

Callbacks used:

* EarlyStopping (patience = 3)
* ModelCheckpoint (best model saved)

---

## 📊 Results (Actual Output)

### 🔹 Validation Accuracy

**≈ 46%**

### 🔹 Classification Report

```
Class  Precision  Recall  F1-score  Support
0      0.44       0.73    0.55      1400
1      0.35       0.38    0.36      1358
2      0.35       0.17    0.23      1400
3      0.53       0.43    0.48      1400
4      0.61       0.56    0.59      1400

Accuracy: 0.46
```

✔ Model performs better on **Class 0 and Class 4**
✖ Significant confusion in **middle severity classes (1–3)**

---

## 🔍 Model Explainability (Grad-CAM)

Grad-CAM is implemented using the **last convolutional layer (`conv2d_2`)** to visualize important retinal regions influencing predictions.

This helps in understanding **where the CNN focuses** while making decisions.

---

## 💾 Model Saving

Model is saved manually after interrupting training:

```python
model.save("custom_dr_model.h5")
```

The model can be reloaded using:

```python
tf.keras.models.load_model("custom_dr_model.h5")
```

---

## ⚠️ Limitations

* No transfer learning
* Low image resolution (128×128)
* CNN struggles with subtle DR features
* Limited performance due to scratch training
* This is not complete trained at this time.
---

## 🔮 Future Improvements

* Use **EfficientNet / ResNet** (transfer learning)
* Increase image size to 224×224
* Apply **Focal Loss**
* Add confusion matrix & ROC curves
* Deploy as a web app

---

## 👤 Author

**Muhammad Jawad Ahsan**
GDG on Campus PIEAS – AI/ML Hackathon Participant

---

## 📜 Disclaimer

This project is for **educational and hackathon purposes only** and is **not intended for clinical use**.
