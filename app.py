import streamlit as st
from PIL import Image
import pandas as pd
from src.predict import predict_image
from src.data_loader import get_data_loaders

# Load classes
_, _, _, classes = get_data_loaders("./data")

st.title("Food-101 Classifier")
st.write("Upload an image to classify the food category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save temp image for prediction
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Predict
    model_path = "./models/resnet18_food101.pth"
    top5_results = predict_image(temp_path, model_path, classes)
    
    # Display top-5 results
    st.write("**Top-5 Predicted Foods**:")
    for i, (food, confidence) in enumerate(top5_results, 1):
        st.write(f"{i}. {food}: {confidence:.2%}")