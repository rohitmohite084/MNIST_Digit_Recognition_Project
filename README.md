# 🧠 MNIST Digit Recognition Project

A high-performance Deep Learning web application that recognizes handwritten digits (0–9) with near-human accuracy. Built using **TensorFlow/Keras** for the neural network and **Streamlit** for the interactive dashboard.

🚀 **Live Demo:** [mnist-digit-recognition-project.streamlit.app](https://mnist-digit-recognition-project.streamlit.app/)

---

## 📌 Overview
This project is an end-to-end Machine Learning application designed to classify handwritten digits from the famous MNIST dataset. It bridges the gap between complex AI models and user-friendly interfaces, allowing anyone to interact with a Convolutional Neural Network (CNN) in real-time.

## ✨ Key Features
* **Interactive Drawing Pad:** Draw digits directly on the web interface using a custom canvas.
* **Image Upload Support:** Upload your own images of handwritten digits for classification.
* **Advanced Preprocessing:** * **Auto-Centering:** Automatically crops and centers the digit for better accuracy.
    * **Contrast Enhancement:** Uses Autocontrast to sharpen the input.
    * **Inversion Logic:** Handles both dark and light background image uploads.
* **Real-time Analytics:** Get instant predictions with confidence scores and probability distribution charts.
* **Validation:** Professional alerts for blank inputs to ensure a smooth user experience.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow, Keras
* **Web Framework:** Streamlit
* **Image Processing:** Pillow (PIL), NumPy
* **Canvas Component:** Streamlit-Drawable-Canvas
* **Deployment:** GitHub & Streamlit Cloud
