from ultralytics import YOLO
import logging
import argparse
import os
import sys

# Basic logging configuration
logging.basicConfig(level=logging.INFO)

def check_positive(value):
    """Ensure the input value is positive for arguments like epochs and batch size."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer")
    return ivalue

def validate_path(path):
    """Validate that the dataset path exists."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Invalid path: {path}. Please provide a valid dataset path.")
    return path

def main():
    # Define the argparse arguments with validation
    parser = argparse.ArgumentParser(description='Train the YOLO model on a dataset')
    parser.add_argument('--data', type=validate_path, required=True, help='Path to the dataset (required)')
    parser.add_argument('--epochs', type=check_positive, default=10, help='Number of epochs (default: 50)')
    parser.add_argument('--batch', type=check_positive, default=16, help='Batch size (default: 16)')
    args = parser.parse_args()

    try:
        model = YOLO('yolo11n-cls.pt')  # Load model
        
        # Start training with specified parameters
        logging.info("Starting training with the following parameters:")
        logging.info(f"Dataset path: {args.data}")
        logging.info(f"Epochs: {args.epochs}")
        logging.info(f"Batch size: {args.batch}")

        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
        )

        # Log the structure of saved runs and results
        logging.info(f"Training completed. Results: {results}")
        logging.info("Training artifacts saved in the following structure:")
        logging.info("runs/classify/train<run_number>/weights/best.pt, last.pt")
        logging.info("runs/classify/train<run_number>/results.png")
        
    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Value error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()