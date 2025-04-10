# 🖼️ CIFAR-10 Image Classification with ResNet50

## 📌 Project Overview
This project implements an image classification pipeline on the CIFAR-10 dataset using a two-phase transfer learning approach. We began with a custom Convolutional Neural Network (CNN), but due to its limitations in capturing complex visual features, we transitioned to a more robust pretrained **ResNet50** architecture. **ResNet50's deep residual learning** allows it to effectively handle diverse image patterns, yielding superior accuracy. The model leverages **ImageNet pretrained weights**, enabling it to benefit from learned features that help improve performance on the CIFAR-10 dataset.


---

## 🎯 Objective
Classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which includes 10 classes ( dog, frog, horse, ship, bird, deer, automobile, airplanes, cats, trucks). The project demonstrates the improvement gained by transitioning from a custom CNN to a fine-tuned ResNet50-based model.

---
## 💡 Why This Project?
This project was built as part of a deep learning portfolio to demonstrate:
- Transfer learning in real-world image classification.
- Practical usage of ResNet50 and fine-tuning.
---

## 🔑 Key Steps

### 📦 1. Importing Required Libraries
- `TensorFlow/Keras` — model building and training
- `OpenCV` — image resizing
- `NumPy` — array manipulation
- `Scikit-learn` — dataset splitting

---

### 🗂️ 2. Dataset Loading & Limiting
- CIFAR-10 is loaded via TensorFlow.
- Limited to **10,000 samples** for quick iteration and computational limitations.

---

### 🧼 3. Preprocessing
- **Resize:** 32×32 → 64×64 via OpenCV → 224×224 using `tf.keras.layers.Resizing`
- **Normalize:** Convert to float32 (no manual scaling—handled by ResNet50 preprocessing)
- **Augmentation:**
  - `RandomFlip`
  - `RandomRotation`
  - `RandomZoom`

---

### 🧪 4. Phase 1: Custom CNN Model
A lightweight CNN was built using:
- Multiple `Conv2D` and `MaxPooling2D` layers
- Dense head with dropout
- Regularization using BatchNormalization and Dropout

> 📉 **Limitation:** Despite reasonable performance, the custom CNN struggled with learning complex patterns, showing reduced generalization.

---

### 🚀 5. Phase 2: Transition to ResNet50
We switched to **ResNet50** because:
- It uses **residual blocks**, allowing for deeper networks and better gradient flow.
- Pretrained on ImageNet — helps transfer general image understanding.
- ResNet50 provides a strong foundation for high-accuracy classification.

---

### 🧱 6.  Building the ResNet50 Model
- Loaded with `include_top=False` and `weights="imagenet"`
- Base layers frozen initially
- Added a **custom classifier head**:
  - `GlobalAveragePooling2D`
  - `Dense → 256 → 128 → 64`
  - `BatchNormalization` after each dense
  - `Dropout(0.6)` for regularization
  - `Dense(10, activation="softmax")` for CIFAR-10

✅ **L2 regularization** applied to all dense layers  
✅ Compiled with:
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam (lr=1e-4)`
- Metric: `accuracy`
- ResNet50 Custom Head Training (before fine-tuning):
The model rapidly improved from **71.50% to 87.9%** validation accuracy in just **10** epochs. The training and validation loss decreased steadily, confirming strong learning and generalization, even before any fine-tuning was applied to the base layers.
---

### 🔁 7. Fine-Tuning ResNet50
- Only `Conv2D` layers unfrozen (kept `BatchNormalization`, `Pooling` layers frozen)
- Reduced learning rate to `1e-5`
- Retrained for another **3 epochs**

---

### 📊 8. Model Performance Comparison

| 🧠 **Model Type** | 🔄 **Training Phase**         | 🎯 **Test Accuracy** | 📉 **Test Loss** |
|-------------------|-------------------------------|----------------------|------------------|
| Custom CNN        | After Fine-tuning             | **75.60%**           | 0.8345           |
| ResNet50          | After Training                | 86.14%               | 3.0072           |
| ResNet50          | After Fine-tuning             | ⭐ **86.81%**        | 2.8772           |

---

## ✅ Conclusion
This project showcases the power of **transfer learning** with **ResNet50** on CIFAR-10:
- Starting with a **custom CNN**, we hit a performance ceiling.
- Transitioning to **ResNet50** dramatically improved accuracy due to deeper layers and pretrained knowledge.
- **Fine-tuning** ResNet50 helped tailor it to CIFAR-10, achieving a final **test accuracy of  86.81%**.

---

## 📁 Files Included
- 📓 [Custom_CNN_CIFAR10_Project.ipynb](./Custom_CNN_CIFAR10_Project.ipynb)
- 📓 `ResNet50_Model.ipynb`: Google Colab notebook for ResNet50 (with fine-tuning)

---

## 💡 Recommendations for Further Improvement

Here are a few ways this project can be extended and improved:

1. **Experiment with Other Architectures**  
   Try models like EfficientNet, DenseNet, or MobileNetV2 for better performance or faster inference.

2. **Integrate MLflow for Experiment Tracking**  
   Track model versions, training metrics, and performance using MLflow.

2. **Use Full CIFAR-10 Dataset**  
   The model is currently trained on a 10K subset; training on the full 50K images can yield even better accuracy.

2. **Data Augmentation Strategies**  
   Explore additional augmentation like brightness, contrast, and CutMix to make the model more robust.

---

## 📦 Requirements
- `TensorFlow: 2.19.0`
- `Keras: 3.8.0`
- `OpenCV: 4.11.0`
- `Scikit-learn: 1.6.1`
- `NumPy: 2.0.2`
---

## 🧠 Author Note
Feel free to explore, experiment on this project! For any questions or suggestions, don’t hesitate to reach out.

