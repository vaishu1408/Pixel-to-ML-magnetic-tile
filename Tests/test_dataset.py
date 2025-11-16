import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dataset import TileSegmentationDataset

dataset = TileSegmentationDataset("data/raw/images")
print("Total samples:", len(dataset))

img, mask = dataset[0]
print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Unique mask values:", mask.unique())
