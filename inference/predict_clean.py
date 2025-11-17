import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms

# allow Python to import from src/
import sys
sys.path.append(".")

from src.unet import UNet


# ---------load trained model---------------

def load_model(model_path, device):
    # build UNet architecture
    model = UNet(in_channels=1, out_channels=1)

    # load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # move model to device
    model.to(device)

    # disable training-specific layers
    model.eval()
    return model


# --------------read & prepare image---------------------

def preprocess_image(img_path):
    # read as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # resize to model input
    img_resized = cv2.resize(img, (256, 256))

    # convert numpy → tensor
    transform = transforms.Compose([
        transforms.ToTensor()  # scale to 0–1 and add channel dim
    ])

    # add batch dimension
    tensor_img = transform(img_resized).unsqueeze(0)
    return img, img_resized, tensor_img



# --------------run model forward pass-------------------------

def predict_mask(model, tensor_img, device):
    # disable gradient tracking
    with torch.no_grad():
        # model output (logits)
        pred = model(tensor_img.to(device))

        # apply sigmoid to convert logits → probability map
        mask = torch.sigmoid(pred).cpu().numpy()[0, 0]

    return mask


# ------------------convert prob map to binary mask-------------------

def postprocess_mask(mask):
    # threshold at 0.5
    bin_mask = (mask > 0.5).astype(np.uint8) * 255
    return bin_mask


# ------------------build overlay image----------------------------

def create_overlay(raw_resized, bin_mask):
    # convert grayscale to RGB
    raw_rgb = cv2.cvtColor(raw_resized, cv2.COLOR_GRAY2RGB)

    # create red defect layer
    red_mask = np.zeros_like(raw_rgb)
    red_mask[:, :, 2] = bin_mask

    # blend raw and red mask
    overlay = cv2.addWeighted(raw_rgb, 0.6, red_mask, 0.4, 0)

    return overlay


# ------------------save outputs--------------------------------

def save_outputs(filename, raw_resized, bin_mask, overlay, save_dir):

    # ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # remove file extension
    base = os.path.splitext(os.path.basename(filename))[0]

    # define paths
    mask_path = f"{save_dir}/mask_{base}.png"
    overlay_path = f"{save_dir}/overlay_{base}.png"
    side_path = f"{save_dir}/side_by_side_{base}.png"

    # save binary mask
    cv2.imwrite(mask_path, bin_mask)

    # save overlay image
    cv2.imwrite(overlay_path, overlay)

    # convert everything to RGB for stacking
    raw_3c = cv2.cvtColor(raw_resized, cv2.COLOR_GRAY2RGB)
    mask_3c = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2RGB)

    # stack raw | mask | overlay horizontally
    side = np.hstack([raw_3c, mask_3c, overlay])
    cv2.imwrite(side_path, side)

    print("\n>>> Saved outputs:")
    print(mask_path)
    print(overlay_path)
    print(side_path)
    print("\n>>> Done!\n")


# ----------------------main execution----------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = p
