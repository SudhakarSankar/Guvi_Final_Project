import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image

# Function to apply sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Function for edge detection
def edge_detect(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Convert to RGB for display

# Function to apply blurring
def blur_image(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# **Improved De-Blurring using Unsharp Masking**
def deblur_image(image):
    gaussian_blur = cv2.GaussianBlur(image, (15, 15), 0)  # Apply Gaussian Blur
    sharpened = cv2.addWeighted(image, 2.0, gaussian_blur, -1.0, 0)  # Unsharp Masking
    return sharpened

# Function to extract text using EasyOCR
def extract_text(image):
    reader = easyocr.Reader(['en'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for OCR
    extracted_text = reader.readtext(gray, detail=0)
    return " ".join(extracted_text)

# Streamlit UI
st.title("📷 Image Processing & OCR App")

# Image Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Show the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ Full-width display

    # Buttons for image processing
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Define variables to store processed images
    processed_image = None
    image_caption = ""
    extracted_text = None

    with col1:
        if st.button("🔪 Sharpen"):
            processed_image = sharpen_image(image)
            image_caption = "Sharpened Image"
    
    with col2:
        if st.button("🖋 Edge Detect"):
            processed_image = edge_detect(image)
            image_caption = "Edge Detected Image"
    
    with col3:
        if st.button("💨 Blur"):
            processed_image = blur_image(image)
            image_caption = "Blurred Image"
    
    with col4:
        if st.button("🎯 De-Blur"):
            processed_image = deblur_image(image)
            image_caption = "De-Blurred Image"
    
    with col5:
        if st.button("📝 Extract Text"):
            extracted_text = extract_text(image)

    # **Display the processed image in full width below the buttons**
    if processed_image is not None:
        st.image(processed_image, caption=image_caption, use_container_width=True)  # ✅ Full-size image

    # **Display extracted text in full width below the image**
    if extracted_text:
        st.markdown("### 📝 Extracted Text:")
        st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; font-size: 18px;'>{extracted_text}</div>", unsafe_allow_html=True)  # ✅ Wide display
