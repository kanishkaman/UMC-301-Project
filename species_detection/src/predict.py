import os
import logging
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

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

def plot_and_save_classification_label(image_path, label, output_dir="../runs/classify/exp"):
  
    # Load the image
    image = Image.open(image_path)
    
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
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    logging.info(f"Image saved to '{output_path}' with classification label: {label}")

    # Optionally display the image
    plt.show()

def run_pipeline(detection_model_path, classification_model_path, image_path):
 
    
    
    image_path = validate_file_path(image_path, file_type="file")
    detection_model_path = validate_file_path(detection_model_path, file_type="file")
    classification_model_path = validate_file_path(classification_model_path, file_type="file")
    validate_image_file(image_path)
    
  
    detection_model = YOLO(detection_model_path)
    classification_model = YOLO(classification_model_path)

    logging.info(f"Running detection on '{image_path}'")
    detection_results = detection_model(image_path)

     # Check if any bounding boxes are detected
    if detection_results[0].boxes:
        logging.info("Objects detected. Running classification...")

        # Run classification model on the image
        classification_results = classification_model.predict(image_path)

        # Get the top classification label for display
        top_class_id = classification_results[0].probs.top1
        top_class_label = classification_results[0].names[top_class_id]
        top_class_conf = classification_results[0].probs.top1conf.item()
        logging.info(f"Top classification label: {top_class_label} with confidence {top_class_conf:.2f}")

        # Plot and save the image with bounding boxes and classification result
        plot_and_save_classification_label(image_path, label=f"{top_class_label} ({top_class_conf:.2f})")
    
    else:
        logging.info("No objects detected. Skipping classification.")
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLO Detection and Classification Pipeline")
    parser.add_argument("--detection_model", type=str, required=True, help="Path to YOLO detection model")
    parser.add_argument("--classification_model", type=str, required=True, help="Path to YOLO classification model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

  
    run_pipeline(args.detection_model, args.classification_model, args.image)
