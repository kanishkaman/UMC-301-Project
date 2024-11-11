import os
import logging
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from siamese_model.siamese import create_siamese_model, load_siamese_model, predict_classification, crop_image

detection_results, croppedimg = crop_image('/Users/pinakinchoudhary/Developer/UMC-301-Project/animal_reidentification/test-zebra-2.jpg', '../model/yolov8n.pt')
model = create_siamese_model(base_model='resnet50', embedding_dim=128)
model = load_siamese_model(model,'/Users/pinakinchoudhary/Developer/UMC-301-Project/model/zebra_siamese.pth')
label, sim = predict_classification("/Users/pinakinchoudhary/Developer/UMC-301-Project/animal_reidentification/test-zebra-2.jpg", model, "zebra_classes")
print(label, " sim= ", sim)
