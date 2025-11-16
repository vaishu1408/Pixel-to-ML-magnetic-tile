import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

# add project root to path so src/ imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dataset import TileSegmentationDataset
from src.unet import UNet

# ---------------------------
# 1. CONFIG
# ---------------------------

# path containing all defect folders (each has .jpg and .png pairs)
ROOT_DIR = "data/raw/images"

BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 20

# select GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------
# 2. LOAD DATA
# ---------------------------

# loads paired image/mask files by filename matching
dataset = TileSegmentationDataset(ROOT_DIR)
print(f"Loaded {len(dataset)} samples.")

# batches data and shuffles each epoch
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# 3. MODEL + LOSS + OPTIMIZER
# ---------------------------

# initialize UNet segmentation model
model = UNet().to(DEVICE)

# binary mask → use BCE loss
criterion = nn.BCELoss()

# Adam optimizer for smoother updates
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------------------
# 4. TRAINING LOOP
# ---------------------------

for epoch in range(1, EPOCHS + 1):

    epoch_loss = 0
    # progress bar for each epoch
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

    for images, masks in progress_bar:
        # move data to chosen device
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # forward pass to get predictions
        preds = model(images)

        # compute pixel-wise BCELoss
        loss = criterion(preds, masks)

        # backprop + update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track loss for monitoring
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch}/{EPOCHS} → Avg Loss: {avg_loss:.4f}")

# ---------------------------
# 5. SAVE TRAINED MODEL
# ---------------------------

# create models directory if missing
os.makedirs("models", exist_ok=True)

# store learned UNet parameters
torch.save(model.state_dict(), "models/unet_magnetic_tile.pth")

print("Model saved successfully: models/unet_magnetic_tile.pth")
