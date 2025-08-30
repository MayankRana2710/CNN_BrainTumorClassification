import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model('brainTumorCNN_model.keras', compile=False)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adamax',
        metrics=['accuracy']
    )
    return model

model = load_cnn_model()

labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("⚙️ Settings")
st.sidebar.info(
    "Upload MRI scans on the main page.\n\n"
    "This CNN model classifies into 4 categories:\n"
    "- Glioma\n- Meningioma\n- No Tumor\n- Pituitary Tumor"
)

st.title("🧠 Brain Tumor Classification with CNN")
st.markdown(
    "This app uses a **Convolutional Neural Network (CNN)** trained on brain MRI scans "
    "to detect different types of tumors. Upload one or more MRI images to get predictions."
)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload MRI Images")
    uploaded_files = st.file_uploader(
        "Upload images (JPG, JPEG, PNG)", 
        accept_multiple_files=True, 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_files:
        images = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((128, 128))
            img_array = np.array(image) / 255.0
            images.append(img_array)
        X = np.array(images)

        st.subheader("🔎 Running Predictions...")
        predictions = model.predict(X)
        predicted_labels = np.argmax(predictions, axis=1)

        for i, uploaded_file in enumerate(uploaded_files):
            label = labels[predicted_labels[i]]
            confidence = np.max(predictions[i]) * 100

            st.success(
                f"**Image {i+1} ({uploaded_file.name}):**\n"
                f"Prediction → **{label}**\n"
                f"Confidence → {confidence:.2f}%"
            )

with col2:
    st.header("Uploaded Images")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, use_container_width=True)
    else:
        st.info("No images uploaded yet.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Developed using Streamlit & TensorFlow"
    "</div>",
    unsafe_allow_html=True
)
