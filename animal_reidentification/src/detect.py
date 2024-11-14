import os
import logging
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from animal_reidentification.siamese_model.siamese import create_siamese_model, load_siamese_model, predict_classification, crop_image

# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

## Aayush Patil
# Configure logging
logging.basicConfig(level=logging.INFO)

def validate_file_path(path, file_type="file"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_type.capitalize()} path '{path}' does not exist.")
    if file_type == "file" and not os.path.isfile(path):
        raise ValueError(f"{file_type.capitalize()} path '{path}' is not a valid file.")
    return path

def validate_image_file(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (IOError, SyntaxError) as e:
        raise ValueError(f"Invalid image file '{image_path}'. Error: {e}")

## Pinakin Choudhary
def run_pipeline_reidentification(detection_model_path, reidentification_model_path, image_path, ref_dir):

    image_path = validate_file_path(image_path, file_type="file")
    detection_model_path = validate_file_path(detection_model_path, file_type="file")
    reidentification_model_path = validate_file_path(reidentification_model_path, file_type="file")
    validate_image_file(image_path)
    
    reidentification_model = create_siamese_model(base_model="resnet50", embedding_dim=128)
    reidentification_model = load_siamese_model(reidentification_model, reidentification_model_path)

    logging.info(f"Running detection on '{image_path}'")
    detection_results, _ = crop_image(image_path, detection_model_path)

    # Load the image
    image = Image.open(image_path)

    # Check if any bounding boxes are detected
    if detection_results is not None:
        logging.info("Objects detected. Running Detection...")

        # Run classification model on the image
        classification_results = predict_classification(image_path, reidentification_model, ref_dir)

        # Get the top classification label and confidence for display
        top_class_label = classification_results[0]
        top_class_similarity = classification_results[1]

        logging.info(f"Top classification label: {top_class_label} with confidence {top_class_similarity:.4f}")

        # Return the image, label, and confidence
        return image, top_class_label, top_class_similarity

    else:
        logging.info("No objects detected. Skipping classification.")
        return image, "No species detected", 0.0