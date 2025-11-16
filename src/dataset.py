import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class TileSegmentationDataset(Dataset):
    def __init__(self, image_root, transform=None):
        """
        image_root: path containing subfolders Blowhole, Crack, Fray, etc.
        Each folder must contain:
            - *.jpg (image)
            - *.png (mask)
        """
        self.image_files = []
        self.mask_files = []
        self.transform = transform

        defect_types = os.listdir(image_root)

        for defect in defect_types:
            defect_folder = os.path.join(image_root, defect, "Imgs")
            if not os.path.isdir(defect_folder):
                continue

            for f in os.listdir(defect_folder):
                if f.endswith(".jpg") or f.endswith(".jpeg"):
                    img_path = os.path.join(defect_folder, f)
                    mask_path = img_path.replace(".jpg", ".png").replace(".jpeg", ".png")

                    if os.path.exists(mask_path):
                        self.image_files.append(img_path)
                        self.mask_files.append(mask_path)

        print(f"Loaded {len(self.image_files)} samples from dataset.")

        # Default transforms
        self.default_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        img = Image.open(img_path).convert("L")     # grayscale
        mask = Image.open(mask_path).convert("L")   # grayscale mask (0/255)

        img = self.default_transform(img)
        mask = self.default_transform(mask)

        # Convert mask to {0,1}
        mask = (mask > 0.5).float()

        return img, mask
