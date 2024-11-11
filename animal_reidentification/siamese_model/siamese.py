import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
import random
from typing import Tuple, Any
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

class SiameseDataset(ImageFolder):
    """
    A dataset class for Siamese Networks using triplet loss.
    Handles classes with single samples through augmentation.
    """
    
    def __init__(self, root: str, transform=None, single_sample_augmentation=True):
        super().__init__(root=root, transform=transform)
        self.single_sample_augmentation = single_sample_augmentation
        
        # Create a dictionary mapping class indices to list of image indices
        self.class_to_indices = {}
        self.single_sample_classes = set()
        
        for idx, (_, class_idx) in enumerate(self.samples):
            if class_idx not in self.class_to_indices:
                self.class_to_indices[class_idx] = []
            self.class_to_indices[class_idx].append(idx)
            
        # Identify classes with single samples
        for class_idx, indices in self.class_to_indices.items():
            if len(indices) == 1:
                self.single_sample_classes.add(class_idx)
                
        # Ensure we have at least 2 classes
        if len(self.class_to_indices) < 2:
            raise ValueError("Dataset must contain at least 2 different classes")

    def augment_single_sample(self, image: Image.Image) -> Image.Image:
        """
        Apply random augmentations to create a positive pair from a single image.
        """
        # List of possible augmentations
        augmentations = [
            # Random horizontal flip
            lambda img: TF.hflip(img) if random.random() > 0.5 else img,
            
            # Random rotation (-10 to 10 degrees)
            lambda img: TF.rotate(img, random.uniform(-10, 10)),
            
            # Random color jitter
            lambda img: TF.adjust_brightness(img, random.uniform(0.8, 1.2)),
            
            # Random crop and resize
            lambda img: TF.resized_crop(
                img,
                top=int(random.uniform(0, img.height * 0.2)),
                left=int(random.uniform(0, img.width * 0.2)),
                height=int(img.height * 0.8),
                width=int(img.width * 0.8),
                size=img.size
            )
        ]
        
        # Apply 2-3 random augmentations
        num_augs = random.randint(2, 3)
        selected_augs = random.sample(augmentations, num_augs)
        
        augmented_image = image
        for aug in selected_augs:
            augmented_image = aug(augmented_image)
            
        return augmented_image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        """
        Returns a triplet of images (anchor, positive, negative)
        Handles single-sample classes through augmentation
        """
        # Get anchor image and its class
        anchor_path, anchor_class = self.samples[index]
        anchor_img = self.loader(anchor_path)
        
        # Handle positive sample selection
        if anchor_class in self.single_sample_classes and self.single_sample_augmentation:
            # For single-sample classes, create positive through augmentation
            positive_img = self.augment_single_sample(anchor_img)
        else:
            # For multi-sample classes, select a different image from same class
            possible_positive_indices = self.class_to_indices[anchor_class].copy()
            possible_positive_indices.remove(index)
            positive_idx = random.choice(possible_positive_indices)
            positive_path, _ = self.samples[positive_idx]
            positive_img = self.loader(positive_path)
        
        # Get negative image (different class than anchor)
        negative_class = random.choice(list(set(self.class_to_indices.keys()) - {anchor_class}))
        negative_idx = random.choice(self.class_to_indices[negative_class])
        negative_path, _ = self.samples[negative_idx]
        negative_img = self.loader(negative_path)
        
        # Apply transformations if specified
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return (anchor_img, positive_img, negative_img), (anchor_class, anchor_class, negative_class)

    def __len__(self) -> int:
        return super().__len__()
    
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Optional, Union, Type

class BaseModelConfig:
    """Configuration for different base models"""
    MODELS = {
        'resnet18': (models.resnet18, 512),
        'resnet34': (models.resnet34, 512),
        'resnet50': (models.resnet50, 2048),
        'resnet101': (models.resnet101, 2048),
        'efficientnet_b0': (models.efficientnet_b0, 1280),
        'efficientnet_b1': (models.efficientnet_b1, 1280),
        'efficientnet_b2': (models.efficientnet_b2, 1408),
        'efficientnet_b3': (models.efficientnet_b3, 1536),
        'vit_b_16': (models.vit_b_16, 768),
        'vit_b_32': (models.vit_b_32, 768),
        'convnext_tiny': (models.convnext_tiny, 768),
        'convnext_small': (models.convnext_small, 768),
        'densenet121': (models.densenet121, 1024),
        'mobilenet_v3_small': (models.mobilenet_v3_small, 576),
        'mobilenet_v3_large': (models.mobilenet_v3_large, 960),
    }

    @classmethod
    def get_model_info(cls, model_name: str) -> Tuple[Type[nn.Module], int]:
        if model_name not in cls.MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_name]

class SiameseNetwork(nn.Module):
    def __init__(
        self, 
        base_model: str = 'resnet50',
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_base: bool = False
    ):
        """
        Siamese Network with configurable base model
        
        Args:
            base_model: Name of the base model to use (see BaseModelConfig.MODELS)
            embedding_dim: Dimension of the output embedding
            pretrained: Whether to use pretrained weights
            freeze_base: Whether to freeze the base model weights
        """
        super(SiameseNetwork, self).__init__()
        
        # Get base model and its output dimension
        model_fn, base_out_dim = BaseModelConfig.get_model_info(base_model)
        base = model_fn(pretrained=pretrained)
        
        # Remove classification head
        if 'resnet' in base_model:
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        elif 'densenet' in base_model:
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        elif 'efficientnet' in base_model:
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        elif 'vit' in base_model:
            self.backbone = nn.Sequential(base.conv_proj, base.encoder)
        elif 'convnext' in base_model:
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        elif 'mobilenet' in base_model:
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        else:
            raise NotImplementedError(f"Model {base_model} structure not implemented")
            
        # Freeze base model if requested
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(base_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.base_model_name = base_model
        
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for one image"""
        x = self.backbone(x)
        
        # Handle different model architectures
        if 'vit' in self.base_model_name:
            x = x[:, 0]  # Use CLS token
        else:
            x = x.view(x.size(0), -1)
            
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
        
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for triplet input"""
        anchor, positive, negative = x
        anchor_embedding = self.forward_one(anchor)
        positive_embedding = self.forward_one(positive)
        negative_embedding = self.forward_one(negative)
        return anchor_embedding, positive_embedding, negative_embedding

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single image"""
        return self.forward_one(x)
    
    def predict_similarity_from_path(self, img1_path, img2_path) -> float:
        """Predict similarity between two images"""
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img1 = transform(img1).unsqueeze(0)
        img2 = transform(img2).unsqueeze(0)
        
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            emb1 = self.get_embedding(img1)
            emb2 = self.get_embedding(img2)
            similarity = F.cosine_similarity(emb1, emb2).item()
        
        return similarity
    
    def predict_similarity(self, img1: Image.Image, img2: torch.Tensor) -> float:
        """
        Predict similarity between two images using the Siamese model
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img1 = transform(img1).unsqueeze(0).to(device)
        
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            emb1 = self.get_embedding(img1)
            emb2 = self.get_embedding(img2)
            similarity = F.cosine_similarity(emb1, emb2).item()
        
        return similarity

    
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def train_siamese_network(
    model: SiameseNetwork,
    train_datasets: List,
    val_datasets: List = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    margin: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Train the Siamese Network
    
    Args:
        model: SiameseNetwork instance
        train_datasets: List of training datasets
        val_datasets: List of validation datasets
        epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Learning rate
        margin: Margin for triplet loss
        device: Device to train on
    
    Returns:
        dict: Training history
    """
    # Combine datasets if multiple are provided
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    if val_datasets:
        val_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Setup training
    model = model.to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    history = {
        'train_loss': [],
        'val_loss': [] if val_datasets else None
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for (anchor, positive, negative), _ in pbar:
            # Move data to device
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            # Forward pass
            anchor_embed, positive_embed, negative_embed = model((anchor, positive, negative))
            
            # Compute loss
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'train_loss': f'{np.mean(train_losses):.4f}'})
        
        epoch_train_loss = np.mean(train_losses)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        if val_datasets:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for (anchor, positive, negative), _ in val_loader:
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    anchor_embed, positive_embed, negative_embed = model((anchor, positive, negative))
                    loss = criterion(anchor_embed, positive_embed, negative_embed)
                    val_losses.append(loss.item())
            
            epoch_val_loss = np.mean(val_losses)
            history['val_loss'].append(epoch_val_loss)
            
            # Update learning rate based on validation loss
            scheduler.step(epoch_val_loss)
            
            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), 'best_siamese_model.pth')
            
            print(f'Epoch {epoch + 1}/{epochs} - '
                  f'Train Loss: {epoch_train_loss:.4f} - '
                  f'Val Loss: {epoch_val_loss:.4f}')
        else:
            print(f'Epoch {epoch + 1}/{epochs} - '
                  f'Train Loss: {epoch_train_loss:.4f}')
    
    return history

# Example usage function
def create_siamese_model(
    base_model: str = 'resnet50',
    embedding_dim: int = 128,
    pretrained: bool = True,
    freeze_base: bool = False
) -> SiameseNetwork:
    """
    Helper function to create a Siamese model with specified configuration
    """
    model = SiameseNetwork(
        base_model=base_model,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        freeze_base=freeze_base
    )
    return model

def load_siamese_model(model: SiameseNetwork, model_path: str, device=device) -> SiameseNetwork:
    """
    Load a saved Siamese model from a file to specified device
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

def crop_image(image_path, model_path):
    """
    Load a saved YOLO model and run predictions on a single image. 
    Also returns the cropped image based on detected bounding boxes.
    
    Parameters:
    - image_path (str): Path to the image for detection.
    - model_path (str): Path to the YOLO model (e.g., 'yolov8n.pt').
    
    Returns:
    - results (list): Prediction results from YOLO.
    - cropped_image (ndarray): Cropped image based on the detected bounding box.
    """
    
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Run prediction on the image
    results = model.predict(image_path)
    
    # Extract the image and bounding boxes from the results
    img = cv2.imread(image_path)  # Read image using OpenCV
    
    # Assuming the first detected object is the one we want to crop (if multiple objects, modify accordingly)
    if results[0].boxes is not None:
        # Get the coordinates of the bounding box (xmin, ymin, xmax, ymax)
        bbox = results[0].boxes[0].xyxy[0].cpu().numpy()  # Get the first bounding box
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # Draw the bounding box on the image
        img_with_bbox = img.copy()
        cv2.rectangle(img_with_bbox, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw bounding box in green
        
        # Crop the image using the bounding box
        cropped_image = img[ymin:ymax, xmin:xmax]  # Crop the image within the bounding box
        
        # # Convert images from BGR to RGB (OpenCV loads images in BGR)
        # img_with_bbox_rgb = cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB)
        # cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        
        # # Show images using matplotlib
        # plt.figure(figsize=(10, 5))

        # # Display the image with bounding box
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_with_bbox_rgb)
        # plt.title("Image with Bounding Box")
        # plt.axis("off")
        
        # # Display the cropped image
        # plt.subplot(1, 2, 2)
        # plt.imshow(cropped_image_rgb)
        # plt.title("Cropped Image")
        # plt.axis("off")
        
        # plt.show()

        # Save cropped image in place of the original image
        cv2.imwrite(image_path, cropped_image)

        return results, cropped_image
    
    else:
        print("No bounding box detected.")
        return None, None

import os
def predict_classification(image_path: str, model: SiameseNetwork, ref_dir: str, species="zebra") -> Any:
    """
    Predict the classification label for an image using the Siamese model
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if species == "zebra":
        # Calculate similarity between the image and zebra reference images in zebra_siamese directory
        zebra_reference_dir = ref_dir
        zebra_reference_images = [f for f in os.listdir(zebra_reference_dir) if f.endswith('.jpg')]
        similarities = []
        for ref_image in zebra_reference_images:
            ref_image_path = os.path.join(zebra_reference_dir, ref_image)
            similarity = model.predict_similarity_from_path(image_path, ref_image_path)
            similarities.append(similarity)

        # Get the label of the most similar zebra reference
        top_idx = np.argmax(similarities)
        top_similarity = similarities[top_idx]
        top_label = zebra_reference_images[top_idx].split('.')[0]
        return top_label, top_similarity
