import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_classifier_model.h5')
    # model = tf.keras.models.load_model('pneumonia_classifier_model')
    return model     

model = load_model()

# Define classes and image size
class_names = ['Normal', 'Pneumonia']
img_height, img_width = 150, 150

# Streamlit UI
st.title('Chest X-Ray Pneumonia Classifier')
st.markdown('Upload a chest X-ray image to get a prediction.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Make prediction on button click
    if st.button('Predict'):
        # Preprocess the image for the model
        img_array = image.img_to_array(image.load_img(uploaded_file, target_size=(img_height, img_width)))
        img_array = np.expand_dims(img_array, axis=0) # Create a batch
        img_array = img_array / 255.0 # Normalize

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        st.write("---")

        # Display the result
        if prediction < 0.5:
            st.success(f"**Prediction:** Normal")
            st.info(f"Confidence: {100 - prediction * 100:.2f}%")
        else:
            st.error(f"**Prediction:** Pneumonia")
            st.info(f"Confidence: {prediction * 100:.2f}%")