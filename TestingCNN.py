import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('brainTumorCNN_model.keras', compile=False)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
    return model

model = load_model()

labels = ['glioma', 'meningioma', 'notumor', 'p']

st.title("Brain Tumor Classification with CNN")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Instructions & Predictions")
    st.write("""
    Upload brain MRI images in JPG, JPEG or PNG format using the panel on the right.
    The model will classify the images into four categories:
    - Glioma
    - Meningioma
    - No Tumor
    - Pituitary Tumor (p)
    """)
    
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
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

        st.subheader(f"Classifying {len(uploaded_files)} image(s)...")
        predictions = model.predict(X)
        predicted_labels = np.argmax(predictions, axis=1)
        
        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"**Image {i+1}:** Predicted class - **{labels[predicted_labels[i]]}**")
            
with col2:
    st.header("Uploaded Images")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, use_container_width=True)
