# this is where I am going to run the u2net_pipleine file, that should:
# 1. Take /raw_images of input images.
# 2. Run U²-Net on them
# 3. Convert the predicted masks to clean binary masks.
# 4. Apply the masks to the original images, producing transparent-background PNGs.
# 5. Save outputs /raw-images folder.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'u2net'))

import glob
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET



# ----------------- Helpers -----------------
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi + 1e-8)


def save_mask(pred, orig_path, output_dir):
    pred = pred.squeeze().cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8) * 255
    mask = Image.fromarray(pred)
    mask.save(os.path.join(output_dir, os.path.basename(orig_path)))


def apply_mask(orig_path, mask_path, output_dir):
    orig = Image.open(orig_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")

    # Resize mask to original image size
    mask = mask.resize(orig.size, resample=Image.BILINEAR)

    alpha = np.array(mask) / 255  # normalize to 0–1
    data = np.array(orig)
    data[..., 3] = (alpha * 255).astype(np.uint8)
    Image.fromarray(data).save(
    os.path.join(output_dir, os.path.splitext(os.path.basename(orig_path))[0] + ".png")
)


# ----------------- Main -----------------
def main(input_dir, output_dir, model_path='u2net/saved_models/u2net/u2net.pth'):
    mask_dir = os.path.join(output_dir, "masks")
    rgba_dir = os.path.join(output_dir, "transparent")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(rgba_dir, exist_ok=True)

    img_list = [
        p for p in glob.glob(os.path.join(input_dir, '*'))
        if p.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"Found {len(img_list)} images in {input_dir}")

    dataset = SalObjDataset(
        img_name_list=img_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = U2NET(3, 1).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    for i, data in enumerate(dataloader):
        print(f"Processing: {os.path.basename(img_list[i])}")

        inputs = data['image'].float().to(device)

        with torch.no_grad():
            d1, *_ = net(inputs)

        pred = normPRED(d1[:, 0, :, :])

        mask_path = os.path.join(mask_dir, os.path.basename(img_list[i]))
        save_mask(pred, img_list[i], mask_dir)
        apply_mask(img_list[i], mask_path, rgba_dir)

        del d1


# ----------------- Run -----------------
if __name__ == "__main__":
    input_dir = os.path.join(os.getcwd(), "raw_images")
    output_dir = os.path.join(os.getcwd(), "masked_images")
    main(input_dir, output_dir)
    print("Background removal complete.")
