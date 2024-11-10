import streamlit as st
from PIL import Image
import os
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from species_detection.langchain.lang import chain
from species_detection.src.detect import run_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
# Streamlit UI

def main():
    st.title("YOLO Object Detection and Classification")

    # Sidebar for navigation
    page = st.sidebar.radio("Select a section", ["Population Trend", "Habitat Mapping", "Species Classification"])

    # Upload detection and classification models
    detection_model_path = "../model/yolov8n.pt"
    classification_model_path = "../model/wildlife.pt"

    if page == "Population Trend":
        st.header("Population Trend Analysis")
        st.write("Here you can analyze the population trend of various species.")
        st.write("This feature is under development.")
    
    elif page == "Habitat Mapping":
        st.header("Habitat Mapping")
        st.write("Here you can explore habitat mapping of species.")
        st.write("This feature is under development.")
    
    elif page == "Species Classification":
        st.header("Species Classification")
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Save the uploaded image temporarily
            temp_image_path = os.path.join(tempfile.gettempdir(), uploaded_image.name)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # Button to run the model
            if st.button("Run Model"):
                # Run the pipeline
                st.write("Processing Image...")
                
                buf, label, confidence = run_pipeline(detection_model_path, classification_model_path, temp_image_path)

                if buf:
                    # Show the image with classification label
                    st.image(buf, caption=f"Prediction: {label} with confidence {confidence:.2f}", use_column_width=True)

                # Show classification result text
                st.write(f"Predicted Class: {label}")
                st.write(f"Confidence: {confidence:.2f}")



                species_info = chain.invoke({"question":label})
                # Display structured information
                st.subheader(f"Information on {label}")
                st.write(species_info)

                # Clean up temporary image file
                os.remove(temp_image_path)

                # Delete the temporary image after inference
                try:
                    os.remove(temp_image_path)
                except Exception as e:
                    st.write(f"Error deleting temporary image: {e}")

if __name__ == "__main__":
    main()