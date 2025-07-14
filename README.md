# ✋ Hand Gesture Recognition with Live Prediction

This project is built as part of SkillCraft's ML series using computer vision and deep learning to recognize hand gestures in real-time via webcam. The model is trained on a custom dataset and predicts gestures live using OpenCV.


## 🔍 Overview

The system captures hand gesture images from a webcam, trains a Convolutional Neural Network (CNN) model on them, and then performs **real-time predictions** using your webcam feed.

### Features:
- Custom CNN model with >99% training accuracy
- Live gesture prediction via webcam
- Supports custom gesture dataset
- Clean and modular code

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Virtual Environment** for package isolation


## 📁 Directory Structure

hand-gesture/
├── data/ # (Ignored) Image dataset for training
├── train_model.py # CNN model training script
├── predict_live.py # Live prediction using webcam
├── gesture_model.h5 # Trained model file (gitignored)
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md


## 🚀 How to Run

### 1. Clone the repo:

git clone https://github.com/praatishthaa/SC_ML_4.git
cd SC_ML_4
2. Create virtual environment:
python -m venv tf-env
3. Activate the environment:
Windows:
.\tf-env\Scripts\activate
Mac/Linux:
source tf-env/bin/activate
4. Install dependencies:
pip install -r requirements.txt
5. Train the model:
python train_model.py
6. Run live prediction:
python predict_live.py
⚠️ Notes
data/ and gesture_model.h5 are excluded via .gitignore for size & privacy.

If you're using a different dataset, make sure it's structured per train_model.py expectations.

Make sure your webcam is accessible and not being used by another application.

🧠 Author
Made with 💻 + ☕ by Praatishthaa
SkillCraft ML Series - Gesture Recognition Project
#SkillCraftTechnology

🌟 Give it a Star!
If this project helped or inspired you, please consider giving it a ⭐ on GitHub.
