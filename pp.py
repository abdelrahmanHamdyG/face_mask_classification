import streamlit as st
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pre-trained model
model = tf.keras.models.load_model('mask_detector.h5')  # Replace 'your_model.h5' with the path to your model file


# Function to preprocess and classify the selected image
def classify_image(image):
    img = np.array(image)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    
    return preds

# Streamlit app
st.title('Image Classification')

# Image picker
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the selected image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to classify the image
    if st.button('Classify'):
        # Perform classification
        label = classify_image(image)
        
        if label[0][0]<label[0][1]:
            st.write(f'No mask {label}')
        else:
            st.write(f"with mask {label}")
