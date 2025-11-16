import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dataset import TileSegmentationDataset
from src.unet import UNet

# ---------------------------
# 1. CONFIG
# ---------------------------
IMAGE_DIR = "data/raw/images"
MASK_DIR = "data/raw/masks"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ---------------------------
# 2. DATA
# ---------------------------
dataset = TileSegmentationDataset(IMAGE_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Loaded {len(dataset)} samples.")

# ---------------------------
# 3. MODEL + LOSS + OPTIM
# ---------------------------
model = UNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------------------
# 4. TRAIN LOOP
# ---------------------------
for epoch in range(1, EPOCHS + 1):

    epoch_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

    for images, masks in progress_bar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch}/{EPOCHS}  → Avg Loss: {epoch_loss/len(loader):.4f}")

# ---------------------------
# 5. SAVE MODEL
# ---------------------------
torch.save(model.state_dict(), "models/unet_magnetic_tile.pth")
print("Model saved successfully: models/unet_magnetic_tile.pth")
