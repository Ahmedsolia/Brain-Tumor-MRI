# 🧠 Brain Tumor MRI Classification - Hybrid AI System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

An end-to-end, production-ready Machine Learning system for classifying Brain Tumors from MRI scans. This project implements a **Hybrid Feature Extraction** approach combining Deep Learning and Texture Analysis, wrapped in an interactive web application with a built-in **Active Learning Feedback Loop**.

## ✨ Key Features

* **🩺 High-Accuracy Classification:** Categorizes MRI scans into 4 distinct classes: `Glioma`, `Meningioma`, `Pituitary`, and `No Tumor`.
* **🛠️ Advanced Preprocessing Pipeline:** Implements robust automated image enhancement, including Median Filtering for noise reduction, CLAHE for contrast equalization, and dynamic **Skull Stripping** to isolate brain tissue.
* **🧬 Hybrid Feature Extraction:** Merges spatial deep-learning features extracted via **ResNet50** (Average Pooling) with statistical texture features utilizing **GLCM** (Gray-Level Co-occurrence Matrix).
* **⚡ Optimized Dimensionality Reduction:** Utilizes **PCA** to compress thousands of features while retaining 95% of the variance, ensuring rapid inference times.
* **📊 Interactive Streamlit Interface:** Features a side-by-side comparison view (Original vs. Preprocessed), dynamic probability bar charts, and downloadable diagnostic reports.
* **🔄 Active Learning / Expert Feedback System:** A built-in data collection pipeline that allows medical professionals to verify or correct model predictions. The system automatically saves the new labeled image and logs the ground truth in a CSV database for future model retraining.

## 🚀 Tech Stack

* **Machine Learning & Deep Learning:** TensorFlow, Keras (ResNet50), Scikit-Learn (PCA, SVM).
* **Computer Vision:** OpenCV, Scikit-Image.
* **Data Manipulation:** NumPy, Pandas.
* **Web UI & Deployment:** Streamlit.

## 📂 Project Structure

```text
Final_project/
├── app.py                      # Main Streamlit application script
├── scaler.pkl                  # Trained StandardScaler artifact (2052 features)
├── pca.pkl                     # Trained PCA artifact
├── svm_resnet_model.pkl        # Trained SVM Classifier
├── requirements.txt            # Python dependencies
├── feedback_log.csv            # Auto-generated database for expert feedback
└── feedback_images/            # Auto-generated directory for storing newly labeled MRI scans
```
### ⚙️ How to Run Locally
1. Clone the repository:

```
git clone [https://github.com/YourUsername/Brain-Tumor-MRI-Classifier.git](https://github.com/YourUsername/Brain-Tumor-MRI-Classifier.git)
cd Brain-Tumor-MRI-Classifier
```
2. Install dependencies:
It is recommended to use a virtual environment.

```
pip install -r requirements.txt
```

2. Launch the application:

```
streamlit run app.py
```
## 📈 Future Improvements
* Continual Learning: Utilize the newly collected images from the feedback_images directory to periodically retrain the SVM and PCA components.

* Model Ensembling: Integrate additional architectures (e.g., EfficientNet, VGG16) and utilize a soft-voting classifier to boost confidence scores on edge cases.

👨‍💻 Developed by: Ahmed Abdelkader
