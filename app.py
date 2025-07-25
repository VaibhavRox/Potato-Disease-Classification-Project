import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title
st.title("🥔 Potato Leaf Disease Classifier")
st.write("Upload a potato leaf image to classify it as **Healthy**, **Early Blight**, or **Late Blight**.")

# Load model (adjust path if needed)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("potatoes.h5")  # Load the newly trained model
    return model

model = load_model()
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # Correct class labels from training

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((256, 256))  # Use your model's expected input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(prediction)

    # Convert class name to user-friendly format
    display_names = {
        'Potato___Early_blight': 'Early Blight',
        'Potato___Late_blight': 'Late Blight', 
        'Potato___healthy': 'Healthy'
    }
    display_class = display_names.get(predicted_class, predicted_class)

    # Show result
    st.success(f"✅ **Prediction:** {display_class}")
    st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")
