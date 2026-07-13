"""
Brain Tumor MRI Classification — Hybrid Model Inference App
=============================================================
Pipeline: Phase I preprocessing -> GLCM texture features + ResNet50 deep
features -> StandardScaler -> PCA -> SVM classifier.

Required artifacts (place alongside this script or set env vars, see below):
    scaler.pkl            StandardScaler fitted on hstack(resnet_features, glcm_features)
    pca.pkl                PCA(n_components=0.95) fitted on the scaled hybrid features
    svm_resnet_model.pkl   SVC(kernel='rbf', probability=True) trained on PCA features

Run with:
    streamlit run app.py
"""

import io
import os

import cv2
import joblib
import numpy as np
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
IMAGE_SIZE = (256, 256)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Artifact locations can be overridden via environment variables so this file
# never needs hardcoded local directories.
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.pkl")
PCA_PATH = os.environ.get("PCA_PATH", "pca.pkl")
SVM_MODEL_PATH = os.environ.get("SVM_MODEL_PATH", "svm_resnet_model.pkl")


# --------------------------------------------------------------------------- #
# Phase I — Preprocessing
# --------------------------------------------------------------------------- #
def apply_median_filter(image, kernel_size=5):
    """Removes salt-and-pepper (speckle) noise while preserving edges."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def apply_manual_histogram_equalization(image):
    """From-scratch histogram equalization (PMF -> CDF -> intensity mapping)."""
    rows, cols = image.shape
    total_pixels = rows * cols

    hist = np.zeros(256)
    for i in range(rows):
        for j in range(cols):
            hist[image[i, j]] += 1

    cdf = np.zeros(256)
    cumulative_sum = 0
    for i in range(256):
        cumulative_sum += hist[i]
        cdf[i] = cumulative_sum

    min_cdf = cdf[cdf > 0].min()
    transform_map = np.zeros(256)
    for i in range(256):
        if cdf[i] > 0:
            transform_map[i] = np.round(((cdf[i] - min_cdf) / (total_pixels - min_cdf)) * 255)

    equalized_image = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            equalized_image[i, j] = transform_map[image[i, j]]

    return equalized_image.astype(np.uint8)


def apply_frequency_filter(image, cutoff_radius=40, order=2):
    """Butterworth Low-Pass Filter applied in the frequency domain (2D FFT)."""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt((U - crow) ** 2 + (V - ccol) ** 2)
    D[D == 0] = 0.01

    mask = 1 / (1 + (D / cutoff_radius) ** (2 * order))
    f_shift_filtered = f_shift * mask

    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back_normalized.astype(np.uint8)


def apply_skull_stripping(image):
    """Otsu's thresholding + morphological opening to isolate brain ROI."""
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)

    brain_mask = np.zeros_like(image)
    cv2.drawContours(brain_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    brain_mask = cv2.dilate(brain_mask, kernel, iterations=1)

    return cv2.bitwise_and(image, image, mask=brain_mask)


def run_phase1_pipeline(gray_image):
    """Runs the full Phase I sequence on a single grayscale image."""
    denoised = apply_median_filter(gray_image)
    enhanced = apply_manual_histogram_equalization(denoised)
    filtered = apply_frequency_filter(enhanced)
    roi_img = apply_skull_stripping(filtered)
    return roi_img


# --------------------------------------------------------------------------- #
# Phase II — Feature Extraction
# --------------------------------------------------------------------------- #
def extract_glcm_features(roi_img):
    """Extracts Contrast, Dissimilarity, Homogeneity, and Energy from GLCM."""
    glcm = graycomatrix(
        roi_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy])


def extract_resnet_features(roi_img, resnet_model):
    """Extracts a 2048-dim deep feature vector using ResNet50 (avg pooling)."""
    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)
    img_array = np.expand_dims(roi_rgb.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array, verbose=0)
    return features.reshape(-1)


def build_feature_vector(gray_image, resnet_model, scaler, pca):
    """
    Full inference-time feature pipeline for a single image:
    preprocess -> GLCM + ResNet50 -> concatenate -> scale -> PCA.
    """
    roi_img = run_phase1_pipeline(gray_image)

    glcm_feats = extract_glcm_features(roi_img)
    resnet_feats = extract_resnet_features(roi_img, resnet_model)

    # Order must match training: ResNet50 features first, then GLCM features.
    hybrid_features = np.hstack((resnet_feats, glcm_feats)).reshape(1, -1)

    scaled_features = scaler.transform(hybrid_features)
    pca_features = pca.transform(scaled_features)
    return pca_features


# --------------------------------------------------------------------------- #
# Model / Artifact Loading (cached so they load only once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading ResNet50 backbone...")
def load_resnet_model():
    return ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        pooling="avg",
    )


@st.cache_resource(show_spinner="Loading trained classifier artifacts...")
def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    return scaler, pca, svm_model


# --------------------------------------------------------------------------- #
# Image Loading Helper
# --------------------------------------------------------------------------- #
def load_uploaded_image_as_grayscale(uploaded_file):
    """Reads an uploaded file object into a resized grayscale numpy array."""
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("L")
    image = image.resize(IMAGE_SIZE)
    return np.array(image)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

    st.title("🧠 Brain Tumor MRI Classifier")
    st.caption("Hybrid Model: ResNet50 Deep Features + GLCM Texture Features + PCA + SVM")
    # -----------------------------------------------------------------
    # Sidebar Help Section
    # -----------------------------------------------------------------
    with st.sidebar:
        st.header("📌 About the System")
        with st.expander("💡 How it Works"):
            st.markdown("""
            This application relies on a **Hybrid AI Model** that goes through 4 fundamental stages to ensure high diagnostic accuracy:

            1. 🛠️ **Image Preprocessing:**
               * **Noise Reduction:** Using a `Median Filter`.
               * **Contrast Enhancement:** To improve detail visibility.
               * **Skull Stripping:** Removing the skull and non-brain tissues so the model focuses strictly on the brain.

            2. 🧠 **Feature Extraction:**
               * Combining deep learning features extracted via `ResNet50` and texture features via `GLCM`.

            3. 📉 **Dimensionality Reduction:**
               * Applying `PCA` to compress thousands of features while retaining 95% of the essential variance for a faster, more efficient diagnosis.

            4. 📊 **Final Classification:**
               * An `SVM` classifier analyzes the compressed data to categorize the MRI scan into one of 4 classes (Glioma, Meningioma, Pituitary, or No Tumor).
               
            ---
            👨‍💻 **Developed by:** Ahmed Abdelkader
            """)
    # -----------------------------------------------------------------

    missing = [p for p in (SCALER_PATH, PCA_PATH, SVM_MODEL_PATH) if not os.path.exists(p)]
    if missing:
        st.error(
            "Missing required model artifact(s): "
            + ", ".join(missing)
            + ". Set SCALER_PATH / PCA_PATH / SVM_MODEL_PATH env vars or place "
            "these files next to app.py."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a brain MRI scan", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        st.info("Upload an MRI image to get a prediction.")
        return

    # -----------------------------------------------------------------
    # Side-by-Side View Feature
    # -----------------------------------------------------------------
    
    # 1. Read the image and convert it to Grayscale
    gray_image = load_uploaded_image_as_grayscale(uploaded_file)
    
    # 2. Apply preprocessing steps to display the result to the user
    with st.spinner("Applying preprocessing filters..."):
        denoised = apply_median_filter(gray_image)
        enhanced = apply_manual_histogram_equalization(denoised)
        filtered = apply_frequency_filter(enhanced)
        preprocessed_image = apply_skull_stripping(filtered)
    
    # 3. Create two equal columns in the UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Original MRI Scan ", use_container_width=True)
        
    with col2:
        # Display the final image after filters and Skull Stripping
        st.image(preprocessed_image, caption="Preprocessed Image ", use_container_width=True)
    
    # -----------------------------------------------------------------

    with st.spinner("Analyzing image..."):
        resnet_model = load_resnet_model()
        scaler, pca, svm_model = load_artifacts()

        gray_image = load_uploaded_image_as_grayscale(uploaded_file)
        feature_vector = build_feature_vector(gray_image, resnet_model, scaler, pca)

        prediction_idx = svm_model.predict(feature_vector)[0]
        predicted_class = CLASS_NAMES[prediction_idx]

        probabilities = None
        if hasattr(svm_model, "predict_proba"):
            probabilities = svm_model.predict_proba(feature_vector)[0]

    st.success(f"Predicted Class: **{predicted_class.upper()}**")

    if probabilities is not None:
        # -----------------------------------------------------------------
        # Probabilities Bar Chart Feature
        # -----------------------------------------------------------------
        st.markdown("### 📊 Prediction Probabilities")
        
        # Prepare the data as a DataFrame for Streamlit to process
        prob_data = pd.DataFrame(
            {
                "Probability (%)": [
                    probabilities[0] * 100,
                    probabilities[1] * 100,
                    probabilities[2] * 100,
                    probabilities[3] * 100,
                ]
            },
            index=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        )
        
        # Display the interactive chart (this will replace the old layout)
        st.bar_chart(prob_data)

    # -----------------------------------------------------------------
    # Download Diagnostic Report Feature
    # -----------------------------------------------------------------
    st.markdown("---") # Separator line for organization
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Prepare report content (ensuring probabilities work correctly inside the file)
    report_content = f"""
    ======================================
         Brain Tumor MRI Classification Report
    ======================================
    
    * Final Prediction: {classes[prediction_idx].upper()}
    
    * Confidence Scores:
    - Glioma: {probabilities[0] * 100:.2f}%
    - Meningioma: {probabilities[1] * 100:.2f}%
    - No Tumor: {probabilities[2] * 100:.2f}%
    - Pituitary: {probabilities[3] * 100:.2f}%
    
    --------------------------------------
    Disclaimer: This is an AI-generated report for preliminary screening purposes only. 
    It is not a substitute for professional medical diagnosis.
    ======================================
    """
    
    # Create the download button
    st.download_button(
        label="📄 Download Diagnostic Report",
        data=report_content,
        file_name="MRI_Diagnostic_Report.txt",
        mime="text/plain"
    )
    # -----------------------------------------------------------------        
    # -----------------------------------------------------------------
    # Save Feedback & Images Feature (Expert Feedback & Data Collection)
    # -----------------------------------------------------------------
    # Save feedback and images with the actual diagnosis (Ground Truth Logging)
    # -----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 👨‍⚕️ Expert Feedback")
    st.write("Help us improve the model by verifying or correcting this prediction.")
    
    def save_detailed_feedback(uploaded_file, prediction, actual_label):
        images_dir = "feedback_images"
        os.makedirs(images_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = uploaded_file.name.split('.')[-1]
        
        # Save the image with a name indicating the prediction and actual diagnosis so it becomes ready-to-use classified data!
        new_image_name = f"img_{timestamp}_pred_{prediction}_actual_{actual_label}.{file_extension}"
        image_path = os.path.join(images_dir, new_image_name)
        
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        csv_file_path = "feedback_log.csv"
        
        # Check if the actual diagnosis matches the prediction
        is_correct = (prediction.lower() == actual_label.lower().replace(" ", ""))
        
        data = [{
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Saved_Image_Name": new_image_name,
            "Model_Prediction": prediction.upper(),
            "Actual_Diagnosis": actual_label.upper(),
            "Status": "Correct" if is_correct else "Incorrect"
        }]
        
        df = pd.DataFrame(data)
        if not os.path.exists(csv_file_path):
            df.to_csv(csv_file_path, index=False)
        else:
            df.to_csv(csv_file_path, mode='a', header=False, index=False)

    # New feedback interface (dropdown menu and submit button)
    disease_options = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Set the default selection to be the same as the model's prediction to save the doctor's time
    default_index = classes.index(predicted_class) 
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # The dropdown list from which the doctor will select the actual diagnosis
        actual_diagnosis = st.selectbox("What is the correct diagnosis?", disease_options, index=default_index)
        
    with col2:
        st.write("") # Add spacing so the button aligns perfectly with the dropdown menu
        st.write("")
        if st.button("💾 Submit Feedback"):
            save_detailed_feedback(uploaded_file, predicted_class, actual_diagnosis)
            
            # Display a thank-you message that changes depending on whether it was correct or incorrect
            if actual_diagnosis.lower().replace(" ", "") == predicted_class.lower():
                st.success("Thank you for verifying! Image and feedback saved.")
            else:
                st.info(f"Feedback recorded. Actual diagnosis marked as: **{actual_diagnosis}**.")
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

if __name__ == "__main__":
    main()