import sys
import os

# Add project root so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import cv2
import torch
import numpy as np

from src.unet import UNet

DEVICE = "cpu"
MODEL_PATH = "models/unet_magnetic_tile.pth"
IMG_SIZE = 256


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Cannot read image: {path}")

    # Resize to model size
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert to tensor (1,1,H,W)
    tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0

    return img_resized, tensor


def overlay_mask(image, mask):
    """Create red overlay on detected defect."""
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    red_mask = np.zeros_like(image_color)
    red_mask[:, :, 2] = mask  # red channel only
    return cv2.addWeighted(image_color, 0.7, red_mask, 0.3, 0)


def main():
    print("\n>>> predict_clean.py started")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    # Prepare output folder
    out_dir = "inference/output"
    os.makedirs(out_dir, exist_ok=True)

    img_name = os.path.basename(args.image)
    img_base = os.path.splitext(img_name)[0]

    print(f">>> Running inference on: {img_name}")

    # Load model
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load image
    raw_img, tensor = load_image(args.image)

    # Predict
    with torch.no_grad():
        pred = model(tensor.to(DEVICE))
        pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

    # Convert to binary mask
    bin_mask = (pred > 0.5).astype(np.uint8) * 255

    # Create overlay
    overlay_img = overlay_mask(raw_img, bin_mask)

    # Convert raw + mask to 3 channels for stacking
    raw_img_color = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    bin_mask_color = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)

    # Build side-by-side image
    side = np.hstack([raw_img_color, bin_mask_color, overlay_img])

    # Output paths
    mask_path = f"{out_dir}/mask_{img_base}.png"
    overlay_path = f"{out_dir}/overlay_{img_base}.png"
    side_by_side_path = f"{out_dir}/side_by_side_{img_base}.png"

    # Save results
    cv2.imwrite(mask_path, bin_mask)
    cv2.imwrite(overlay_path, overlay_img)
    cv2.imwrite(side_by_side_path, side)

    print("\n>>> Saved outputs:")
    print(mask_path)
    print(overlay_path)
    print(side_by_side_path)
    print("\n>>> Done!\n")


if __name__ == "__main__":
    main()
