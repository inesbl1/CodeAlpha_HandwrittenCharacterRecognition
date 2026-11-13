# Handwritten Character Recognition — CodeAlpha ML Internship (Task 3)

Recognizing handwritten digits (0–9) with a **Convolutional Neural Network (CNN)** built in TensorFlow/Keras, plus an interactive demo where you can draw a digit and get a live prediction.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D9ERW6A5puGd_xM3_0z5VMcKsXXTUaBt)

## Objective
Identify handwritten characters from images using image processing and deep learning.

## Dataset
**MNIST**: 70,000 grayscale 28×28 images of handwritten digits (60,000 train / 10,000 test), with 10 balanced classes.

## Approach
1. **Preprocessing**: pixel normalization to [0, 1] and reshaping to `(28, 28, 1)`
2. **Data augmentation**: random rotation, zoom and translation (training only)
3. **CNN architecture**:
   ```
   [Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)]
   [Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)]
   Flatten → Dense(128) → Dropout(0.5) → Dense(10, softmax)
   ```
4. **Training**: Adam optimizer, sparse categorical cross-entropy, `EarlyStopping` + `ReduceLROnPlateau`
5. **Evaluation**: test accuracy, per-class precision/recall/F1, confusion matrix, error analysis
6. **Demo**: Gradio sketchpad that preprocesses the drawing the same way as MNIST (crop, resize to 20×20, center in 28×28) and shows the 28×28 image seen by the model

## Results
| Metric | Score |
|---|---|
| Test accuracy | **98.79 %** |
| Misclassified images | 121 / 10,000 |
| Macro F1-score | 0.988 |

## How to run
**Google Colab (recommended):** click the badge above, select *Runtime → Change runtime type → GPU*, then *Run all*.
The last cell launches the drawing demo with a public link.

**Locally:**
```bash
git clone https://github.com/inesbl1/CodeAlpha_HandwrittenCharacterRecognition.git
cd CodeAlpha_HandwrittenCharacterRecognition
pip install -r requirements.txt
jupyter notebook Handwritten_Character_Recognition.ipynb
```

## Tech stack
Python · TensorFlow / Keras · NumPy · SciPy · scikit-learn · Matplotlib · Seaborn · Pillow · Gradio

## Future work
- Extend to **EMNIST** letters (A–Z) by changing the output layer to 26 classes
- Recognize full words or sentences with a **CRNN** (CNN + LSTM + CTC loss)
- Deploy the demo on Hugging Face Spaces

---
Made by **Ines BOUALI** during the **CodeAlpha Machine Learning Internship** · [LinkedIn](https://www.linkedin.com/in/ines-bouali-6b4b69311)