import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dataset import TileSegmentationDataset
dataset = TileSegmentationDataset("data/raw/images")

print("Pairs:", len(dataset))

for i in range(10):
    print("\nPAIR", i)
    print("IMG :", dataset.image_files[i])
    print("MASK:", dataset.mask_files[i])
