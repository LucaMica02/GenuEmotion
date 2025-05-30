# 🎓 Final Project – AI Lab Course

## 🤖 Emotion Recognition & Genuineness Detection

**Using Computer Vision and Machine Learning**

### 📌 Project Description

This project aims to develop a system that can:

- 🔍 Recognize facial emotions (e.g., happy, sad, angry, surprised)
- ✅ Detect the authenticity (genuineness) of those emotions

Built with the power of **computer vision** and **machine learning**, this tool explores both the surface expression and the truth behind it.

### 👨‍💻 Technologies Used

- Python
- OpenCV
- PyTorch
- OpenFace

### 📁 Structure

- **`emotion_recognition/`**  
  Contains all the code related to the emotion recognition task:

  - Model architecture
  - Dataset loading and preprocessing
  - Training and evaluation scripts

- **`genuineness_detection/`**  
  Contains the code for the genuineness (authenticity) detection task:

  - Neural network model
  - Micro-expression dataset integration
  - Training and evaluation pipeline

- **`main.py`**  
  The main entry point of the project.  
  Use this script to:
  - Run real-time inference using your webcam
  - Load and test the pretrained models for emotion recognition and genuineness detection

> 💡 Make sure your **webcam** is connected and all **dependencies** are installed before running `main.py`.  
> 💡 Remember to **train the models** before running `main.py`.  
> 💡 If you want to get predictions based on **Action Units (AUs)**, you need to install and use the **OpenFace** tool.  
> 💡 Don't forget to **update the file paths (PATHS)** in the code according to your local environment.

### 📂 Dataset Access

You can access the original datasets used in this project via the following link:

👉 [Download Dataset from Google Drive](https://drive.google.com/drive/u/0/folders/1w5s0F2jjYb_B5ut2sYBO2ZqEqnOz0aLu)

### 📈 Goals

- Train an emotion classification model
- Integrate a veridicality (truthfulness) classifier
- Provide real-time inference with a simple UI
