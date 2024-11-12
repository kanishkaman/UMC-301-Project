import os
import logging
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_pipeline(detection_model_path, classification_model_path, image_path):
    detection_model = YOLO(detection_model_path)
    classification_model = YOLO(classification_model_path)

    logging.info(f"Running detection on '{image_path}'")
    detection_results = detection_model(image_path)

    # Load the image
    image = Image.open(image_path)

    # Check if any bounding boxes are detected
    if detection_results[0].boxes:
        logging.info("Objects detected. Running classification...")

        # Run classification model on the image
        classification_results = classification_model.predict(image_path)

        # Get the top classification label and confidence for display
        top_class_id = classification_results[0].probs.top1
        top_class_label = classification_results[0].names[int(top_class_id)]
        top_class_conf = classification_results[0].probs.top1conf.item()

        logging.info(f"Top classification label: {top_class_label} with confidence {top_class_conf:.2f}")

        # # Plot and save the image with classification result
        # plot_and_save_classification_label(image, label=f"{top_class_label} ({top_class_conf:.2f})")

        # Return the image, label, and confidence
        return image, top_class_label, top_class_conf

    else:
        logging.info("No objects detected. Skipping classification.")
        return image, "No species detected", 0.0

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="YOLO Detection and Classification Pipeline")
#     parser.add_argument("--detection_model", type=str, required=True, help="Path to YOLO detection model")
#     parser.add_argument("--classification_model", type=str, required=True, help="Path to YOLO classification model")
#     parser.add_argument("--image", type=str, required=True, help="Path to the image file")
#     args = parser.parse_args()

#     image, label, confidence = run_pipeline(args.detection_model, args.classification_model, args.image)
#     print(f"Predicted Class: {label} with Confidence: {confidence:.2f}")
