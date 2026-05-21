import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2

# =============================================================================
# 1. CUSTOM DATASET FOR MUJOCO SPATIAL DATA
# =============================================================================
class SpatialControlDataset(Dataset):
    """
    Loads edge-mapped images and combines them with noisy absolute XY coordinate priors
    to serve as inputs, mapping them to local End-Effector relative XYZ+QUAT labels.
    """
    def __init__(self, dataset_path, transform=None):
        self.data_dir = os.path.dirname(dataset_path)
        # Load the saved metadata array (handles object arrays from np.save)
        self.records = np.load(dataset_path, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Load image (Edge maps are saved as grayscale/single-channel PNGs)
        img_path = os.path.join(self.data_dir, record["image_file"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Failed to load image asset at: {img_path}")
            
        # Normalize image pixel values to [0.0, 1.0] and add channel dimension: (1, H, W)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Inputs & Labels
        spatial_prior = record["input_noisy_xy"].astype(np.float32)  # Shape: (2,)
        relative_target = record["label_relative_xyzw"].astype(np.float32) # Shape: (7,) [x, y, z, qx, qy, qz, qw]
        
        # Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(image)
        spatial_tensor = torch.from_numpy(spatial_prior)
        label_tensor = torch.from_numpy(relative_target)
        
        return (image_tensor, spatial_tensor), label_tensor


# =============================================================================
# 2. MULTI-MODAL VISUAL HEAD NETWORK ARCHITECTURE
# =============================================================================
class VisualSpatialHead(nn.Module):
    """
    A Multi-Modal architecture that processes structural edge topologies via a 
    CNN backbone, infuses an absolute workspace spatial prior, and regresses 
    relative transformations (3D translation + 4D quaternion orientation).
    """
    def __init__(self, spatial_prior_dim=2, output_dim=7):
        super(VisualSpatialHead, self).__init__()
        
        # Convolutional Feature Extractor (Input: 1 x 320 x 320)
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # -> 16 x 160 x 160
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32 x 80 x 80
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64 x 40 x 40
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 128 x 20 x 20
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> 256 x 10 x 10
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((4, 4)) # Forces spatial bottleneck down to a reliable fixed footprint -> 256 x 4 x 4
        )
        
        # Visual feature flattening calculation: 256 channels * 4 * 4 feature grid = 4096
        num_flattened_features = 256 * 4 * 4
        
        # Fully Connected Fusion Layers
        self.fusion_regressor = nn.Sequential(
            nn.Linear(num_flattened_features + spatial_prior_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Enhances generalization against simulator variations
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, output_dim) # Outputs raw continuous values: [X, Y, Z, qx, qy, qz, qw]
        )

    def forward(self, image, spatial_prior):
        # Extract visual feature map matrix
        visual_features = self.cnn_backbone(image)
        visual_features = torch.flatten(visual_features, start_dim=1)
        
        # Late-Fusion: Concatenate visual features with the low-dimensional structural prior vector
        fused_vector = torch.cat((visual_features, spatial_prior), dim=1)
        
        # Regress spatial coordinates
        predictions = self.fusion_regressor(fused_vector)
        return predictions


# =============================================================================
# 3. TRAINING LOOP PIPELINE UTILITY
# =============================================================================
def train_pipeline(dataset_npy_path, epochs=30, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using runtime device platform execution: {device}")
    
    # Instantiate dataset processing
    dataset = SpatialControlDataset(dataset_npy_path)
    
    # Train / Validation Split (80% / 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize network, loss function (MSE is standard for regression), and Adam Optimizer
    model = VisualSpatialHead().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    print(f"Dataset Size: {len(dataset)} | Train Steps: {len(train_loader)} batches/epoch")
    print("====== Initiating Network Training Framework ======")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        
        for (images, priors), labels in train_loader:
            images = images.to(device)
            priors = priors.to(device)
            labels = labels.to(device)
            
            # Forward execution
            optimizer.zero_grad()
            predictions = model(images, priors)
            loss = criterion(predictions, labels)
            
            # Backpropagation updates
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        
        # Validation Eval Evaluation Check
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for (images, priors), labels in val_loader:
                images = images.to(device)
                priors = priors.to(device)
                labels = labels.to(device)
                
                predictions = model(images, priors)
                loss = criterion(predictions, labels)
                running_val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] -> Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
        # Save checkpoints safely
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(os.path.dirname(dataset_npy_path), "best_visual_head.pth"))
            print("  ↳ Saved improved network checkpoint weight matrix configuration state.")
            
    print("===================================================")

if __name__ == "__main__":
    # Point directly to your generated numpy tracking record log stack
    target_data_file = os.path.expanduser("~/RL_Proj/collected_data/visual_training_dataset.npy")
    if os.path.exists(target_data_file):
        train_pipeline(target_data_file, epochs=25, batch_size=16)
    else:
        print(f"Execution Error: Synthetic sample matrix file target not found at {target_data_file}.\n"
              f"Please verify your generator output folder settings.")