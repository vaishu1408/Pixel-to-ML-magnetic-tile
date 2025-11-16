import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from src.unet import UNet

MODEL_PATH = "models/unet_magnetic_tile.pth"

def load_image(path):
    img = Image.open(path).convert("L").resize((256, 256))
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # shape: [1,1,H,W]
    return img

def predict(image_path):
    device = "cpu"

    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    img = load_image(image_path).to(device)

    with torch.no_grad():
        pred = model(img)
        pred = (pred > 0.5).float()

    mask = pred.squeeze().cpu().numpy() * 255
    cv2.imwrite("prediction_output.png", mask)
    print("Saved prediction_output.png")

predict("data/raw/images/MT_Crack/Imgs/exp1_num_249594.jpg")
