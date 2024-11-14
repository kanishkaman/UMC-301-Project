# Animal Re-Identification Using Siamese Network

## Overview

This project implements an animal re-identification pipeline focused on identifying individual zebras from wildlife images. The solution combines YOLOv8, OpenCV, and a PyTorch-based Siamese Network to detect and re-identify zebras. An interactive Streamlit interface enables easy usability.

## Key Features

1. **Wildlife Dataset**
    - Utilizes the wildlife-dataset repository to import zebra data for training and testing.
2. **Object Detection**
    - Leverages YOLOv8 to detect zebras in images and generate bounding boxes for cropped inputs.
3. **Bounding Box Cropping**
    - Uses OpenCV to extract the bounding boxes of detected zebras for subsequent identification.
4. **Re-Identification Model**
    - Built with PyTorch, the Siamese Network is trained to distinguish between individual zebras.
    - **Architecture:**
      - The network consists of two identical sub-networks (shared weights) that extract feature embeddings from input images.
      - Outputs are compared using a distance metric (e.g., Euclidean or cosine similarity) to evaluate their similarity.
    - **Triplet Loss:**
      - The loss function ensures that:
         - The distance between an anchor image and a positive (same individual) is minimized.
         - The distance between an anchor image and a negative (different individual) is maximized.
      - This helps in clustering embeddings of the same zebra while separating embeddings of different zebras.
    - **Single Sample Augmentation:**
      - In the SiameseDataset class, augmentation techniques (e.g., flipping, cropping, or color jitter) are applied to expand limited training data.

## Streamlit Integration

1. **Zebra Classes Folder**
    - Contains a folder named `zebra_classes`, with one reference image for every known zebra.
2. **Image Matching**
    - The uploaded zebra image is processed through the Siamese Network to generate its feature embedding.
    - Embeddings are compared against those of the reference images in the `zebra_classes` folder.
3. **Threshold for New Zebras**
    - If the maximum similarity score is below a predefined threshold, the system classifies the uploaded image as a new, unregistered zebra.

## Requirements

To replicate this project, install the following dependencies:

- Python 3.8+
- PyTorch
- YOLOv8 (via Ultralytics)
- OpenCV
- Streamlit
- numpy, pandas, and other common libraries

## How to Use

1. **Setup**
    - Clone this repository and ensure access to the wildlife-dataset repository.
    - Organize zebra reference images in the `zebra_classes` folder, ensuring one image per individual zebra. (Has already been done for testing purposes.)
2. **Run Streamlit App**
    - Launch the app using:
      ```bash
      streamlit run app.py
      ```
    - Upload a zebra image to identify it or classify it as a new individual.
3. **Customization**
    - Modify the similarity threshold in the code to adjust sensitivity for new zebra detection.
    - Add more reference images to the `zebra_classes` folder for a larger dataset.
