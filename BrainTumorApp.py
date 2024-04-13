import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import warnings
import time
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create a Streamlit theme using the defined colors
st.set_page_config(
    page_title="Brain Detection App",
    page_icon="🧠",  
    layout="wide",
    initial_sidebar_state="auto",
)



@st.cache
def preprocess_image(image):
    """
    Preprocess the input image to match the model's expected input.
    This includes resizing, converting to grayscale, applying Gaussian blur,
    thresholding, and finding the largest contour to crop the brain area.
    Args:
        image: A PIL Image instance.
    Returns:
        preprocessed_image: The preprocessed image suitable for the model.
    """
    # Convert the PIL Image to a numpy array
    image = np.array(image)

    # Resize the image to 240x240
    image = cv2.resize(image, (240, 240))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours and get the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points and crop the image
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    # Resize to ensure the input size is consistent with the model's expected input
    preprocessed_image = cv2.resize(new_image, (240, 240))

    # Normalize the image
    preprocessed_image = preprocessed_image / 255.0

    # Add a dimension to match the model's input shape (batch size dimension)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    return preprocessed_image

# Load the trained model (Ensure the path is correct and accessible)
model_path = 'brain_tumor_detector.h5'
model = load_model(model_path)

# Define the prediction function
@st.cache
def predict(image):
    """
    Predict whether the MRI image has a brain tumor.
    Args:
        image: The preprocessed image suitable for the model.
    Returns:
        prediction: The model's prediction.
    """
    prediction = model.predict(image)
    return prediction

#---------------UI------------------------

# Streamlit UI
st.title("Brain Tumor Detection App")
st.write("Welcome to the Brain Detection App! 🧠")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["png", "jpg", "jpeg"])

# Initialize prediction
prediction = None

if uploaded_file is not None:
    try:
        # Display the uploaded image
        uploaded_image = Image.open(uploaded_file)
        st.sidebar.image(uploaded_image, caption='Uploaded MRI', use_column_width=True)
        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(uploaded_image)
        # Make a prediction on the preprocessed image
        if st.sidebar.button("Make Prediction"):
            prediction = predict(preprocessed_image)
            with st.spinner('Please wait the model is predicting ﮩـﮩﮩ٨ـ🫀ﮩ٨ـﮩﮩ٨ـ...'):
                time.sleep(5)
            st.success('Done!')

        # Display the prediction
        if prediction is not None and prediction[0][0] > 0.5:
            st.write("The model predicts the presence of a brain tumor. 😢")
        elif prediction is not None:
            st.write("The model predicts no brain tumor is present. 😃")
    except Exception as e:
        st.write("Error Please Upload an MRI image of Brain Tumor :", str(e))
