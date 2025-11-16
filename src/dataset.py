import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TileSegmentationDataset(Dataset):
    def __init__(self, image_root):
        """
        image_root: path to data/raw/images
                    Contains defect folders, each with .jpg and .png mixed
        """

        self.image_files = []
        self.mask_files = []

        # loop through defect classes
        for defect in os.listdir(image_root):
            defect_path = os.path.join(image_root, defect, "Imgs")
            if not os.path.isdir(defect_path):
                continue

            # for each jpg, find matching png
            for f in os.listdir(defect_path):
                if not f.lower().endswith(".jpg"):
                    continue

                base = f[:-4]  # remove ".jpg"
                image_path = os.path.join(defect_path, f)
                mask_path = os.path.join(defect_path, base + ".png")

                if os.path.exists(mask_path):
                    self.image_files.append(image_path)
                    self.mask_files.append(mask_path)

        print(f"Loaded {len(self.image_files)} correctly paired samples.")

        self.transform_img = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.transform_mask = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx])
        mask = Image.open(self.mask_files[idx])

        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()

        return img, mask
