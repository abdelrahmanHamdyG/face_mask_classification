import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


def preprocess_image(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


model = tf.keras.models.load_model("mask_classifier.h5")


def predict(image):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    return predictions


def main():
    st.title("Image Classifier")
    st.write("Upload an image and I'll predict its content.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:

        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)


        if st.button('Predict'):
            with st.spinner('Predicting...'):
                predictions = predict(uploaded_image)

            st.write(predictions)

            st.write('Predictions:')
            if predictions>=0.5:
                st.write("there is a mask")
            else:
                st.write("no mask")

if __name__ == "__main__":
    main()
