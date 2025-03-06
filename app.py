import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import plotly.graph_objects as go
from PIL import Image
from time import sleep


# Load and preprocess the image
def model_predict(image_path):
    try:
        model = tf.keras.models.load_model(r"E:\streamlit\best_model (2).keras")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

    try:
        img = cv2.imread(image_path)  # Read the file and convert it into an array
        if img is None:
            raise ValueError("Unable to load image. Please check the file format.")
    except Exception as e:
        st.error(f"Error reading the image: {e}")
        return None, None

    try:
        H, W, C = 224, 224, 3
        img = cv2.resize(img, (H, W))  # Resize image to match model input size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = np.array(img)
        img = img.astype("float32")
        img = img / 255.0  # Rescaling
        img = img.reshape(1, H, W, C)  # Reshaping image

        # Get prediction probabilities
        prediction_probs = model.predict(img)[0]
        result_index = np.argmax(prediction_probs)
        confidence = prediction_probs[result_index] * 100  # Confidence percentage
    except Exception as e:
        st.error(f"Error during image preprocessing or prediction: {e}")
        return None, None

    return result_index, confidence


# Sidebar
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Main Page
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center;'>Welcome to the Future of Plant Health!</h1>",
        unsafe_allow_html=True,
    )

    # Background Animation (CSS)
    st.markdown(
        """
        <style>
            @keyframes move {
                0% {background-position: 0% 0%;}
                50% {background-position: 100% 100%;}
                100% {background-position: 0% 0%;}
            }

            body {
                animation: move 10s infinite linear;
                background: linear-gradient(45deg, #a8e063, #56ab2f);
                background-size: 200% 200%;
                height: 100vh;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.image("E:\\streamlit\\UI.png", width=600)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")

    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        # Define the save path
        save_path = os.path.join(os.getcwd(), test_image.name)
        try:
            with open(save_path, "wb") as f:
                f.write(test_image.getbuffer())
        except Exception as e:
            st.error(f"Error saving the image: {e}")

    if st.button("Show Image"):
        st.image(test_image, width=400, use_column_width=True)

    # Predict button with creative enhancements
    if st.button("Predict"):
        with st.spinner("Analyzing Image..."):
            sleep(2)  # Simulate prediction delay
            result_index, confidence = model_predict(save_path)

        if result_index is not None:
            class_name = [
                "Apple___Apple_scab",
                "Apple___Black_rot",
                "Apple___Cedar_apple_rust",
                "Apple___healthy",
                "Blueberry___healthy",
                "Cherry_(including_sour)___Powdery_mildew",
                "Cherry_(including_sour)___healthy",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                "Corn_(maize)___Common_rust_",
                "Corn_(maize)___Northern_Leaf_Blight",
                "Corn_(maize)___healthy",
                "Grape___Black_rot",
                "Grape___Esca_(Black_Measles)",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "Grape___healthy",
                "Orange___Haunglongbing_(Citrus_greening)",
                "Peach___Bacterial_spot",
                "Peach___healthy",
                "Pepper,_bell___Bacterial_spot",
                "Pepper,_bell___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
                "Raspberry___healthy",
                "Soybean___healthy",
                "Squash___Powdery_mildew",
                "Strawberry___Leaf_scorch",
                "Strawberry___healthy",
                "Tomato___Bacterial_spot",
                "Tomato___Early_blight",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot",
                "Tomato___Spider_mites Two-spotted_spider_mite",
                "Tomato___Target_Spot",
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato___Tomato_mosaic_virus",
                "Tomato___healthy",
            ]

            # Create a pie chart using Plotly
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            "Predicted Class",
                            "Confidence Level",
                            "Other Predictions",
                        ],
                        values=[confidence, 100 - confidence, 5],
                        hoverinfo="label+percent",
                        textinfo="percent+label",
                        title=f"Prediction: {class_name[result_index]}",
                    )
                ]
            )

            # Show pie chart
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Prediction failed. Please check the logs for more details.")
