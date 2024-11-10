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

from species_detection.src.detect import run_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to validate file paths
def validate_file_path(path, file_type="file"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_type.capitalize()} path '{path}' does not exist.")
    if file_type == "file" and not os.path.isfile(path):
        raise ValueError(f"{file_type.capitalize()} path '{path}' is not a valid file.")
    return path

# Function to validate the image file
def validate_image_file(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (IOError, SyntaxError) as e:
        raise ValueError(f"Invalid image file '{image_path}'. Error: {e}")

# Function to plot and save the classification label
def plot_and_save_classification_label(image, label, output_dir="../runs/classify/exp"):
    # Set up matplotlib figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Annotate the image with classification label
    ax.text(
        10, 20, label, color='red', fontsize=16,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
    )

    # Hide axes
    plt.axis('off')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    output_path = os.path.join(output_dir, "classified_" + os.path.basename(image.filename))
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    logging.info(f"Image saved to '{output_path}' with classification label: {label}")

    # Convert figure to BytesIO object for Streamlit
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return buf

# Streamlit UI
def main():
    st.title("YOLO Object Detection and Classification")

    # Sidebar for navigation
    page = st.sidebar.radio("Select a section", ["Population Trend", "Habitat Mapping", "Species Classification"])

    # Upload detection and classification models
    detection_model_path = "/home/guest_quest_70/UMC_Project_Group_5/UMC-301-Project/model/yolov8n.pt"
    classification_model_path = "/home/guest_quest_70/UMC_Project_Group_5/UMC-301-Project/model/wildlife.pt"

    if page == "Population Trend":
        st.header("Population Trend Analysis")
        st.write("Here you can analyze the population trend of various species.")
        # You can add the population trend related logic here
        # E.g., display a graph or some other relevant content
        st.write("This feature is under development.")
    
    elif page == "Habitat Mapping":
        st.header("Habitat Mapping")
        st.write("Here you can explore habitat mapping of species.")
        # You can add habitat mapping related logic here
        # E.g., displaying a map or information about species' habitats
        st.write("This feature is under development.")
    
    elif page == "Species Classification":
        st.header("Species Classification")
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Save the uploaded image temporarily
            temp_image_path = os.path.join(tempfile.gettempdir(), uploaded_image.name)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            # Run the pipeline
            st.write("Processing Image...")

            buf, label, confidence = run_pipeline(detection_model_path, classification_model_path, temp_image_path)

            if buf:
                # Show the image with classification label
                st.image(buf, caption=f"Prediction: {label} with confidence {confidence:.2f}", use_column_width=True)

            # Show classification result text
            st.write(f"Predicted Class: {label}")
            st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
