import torch
import numpy as np
import cv2
from pathlib import Path
import time
from unet import UNet   # model của bạn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def normalize_wafer(img):
    # img = img - 0.22888532845
    p1 = np.percentile(img, 0.1)
    p99 = np.percentile(img, 99.9)
    if p1==p99:
        p1=img.min()
        p99=img.max()
    img = (img - p1) / (p99 - p1)
    img = np.clip(img, 0, 1)

    return img

def normalize_wafer_min_max(img):
    # img = img - 0.22888532845
    
    p1=img.min()
    p99=img.max()
    img = (img - p1) / (p99 - p1)
    img = np.clip(img, 0, 1)

    return img

def normalize_wafer_1(img,p1, p99):
    
    img = (img - p1) / (p99 - p1)
    img = np.clip(img, 0, 1)

    return img

def load_image(img_path):

    # img = cv2.imread(img_path)[:,:,0]/255.0
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)/65535

    if img is None:
        raise RuntimeError("Cannot load image")

    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    img = normalize_wafer(img)

    img = torch.from_numpy(img).float().unsqueeze(0)

    return img

def load_image_1(img_path,p1,p99):

    # img = cv2.imread(img_path)[:,:,0]/255.0
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)/65535

    if img is None:
        raise RuntimeError("Cannot load image")

    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    img = normalize_wafer_1(img,p1/65535,p99/65535)

    img = torch.from_numpy(img).float().unsqueeze(0)

    return img

def predict_img(model, img_tensor):

    img_tensor = img_tensor.to(device)

    with torch.no_grad():

        residual = model(img_tensor)
        
        # enhanced = img_tensor + residual
        enhanced = img_tensor*(1 + residual)
        print(enhanced)

    enhanced = enhanced.squeeze().cpu().numpy()

    enhanced = np.clip(enhanced, 0, 1)

    return enhanced


def save_image(img, path):

    img = (img * 65535).astype(np.uint16)

    cv2.imwrite(path, img)


# def main():

#     model_path = "checkpoints/checkpoint_epoch50.pth"
#     input_image = "test_image.tif"
#     output_image = "enhanced.tif"

#     model = UNet(n_channels=1, n_classes=1)

#     model.load_state_dict(torch.load(model_path, map_location=device))

#     model.to(device)
#     model.eval()

#     img = load_image(input_image)

#     enhanced = predict_img(model, img)

#     save_image(enhanced, output_image)

#     print("Saved enhanced image:", output_image)


# if __name__ == "__main__":
#     main()

# model_path = "checkpointsl21040/checkpoint_epoch50.pth"
# model_path = "savecheckpoints/checkpointsmerge/checkpoint_epoch50.pth"
# model_path = "savecheckpoints/checkpoints/checkpoint_epoch50.pth"
# model_path = "savecheckpoints/checkpoints20260401/checkpoint_epoch50.pth"
# model_path = "savecheckpoints/checkpoints20260420/checkpoint_epoch50.pth"

model_path = "savecheckpoints/checkpoints20260409/checkpoint_epoch20.pth"

model = UNet(n_channels=1, n_classes=1)

model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()

# input_image = r"G:\AutoImageProcessing\Slot0113.tif"

# input_image = r"G:\AutoImageProcessing\WaferData\Image2\Invert_20260323053237\Branch_2_4.tif"
# output_image = "enhanced.tif"
# # img = torch.from_numpy(np.load(input_image)).float().unsqueeze(0)#
# img = load_image_1(input_image,29135,65535)
# # img = load_image_1(input_image,54000,60000)
# # img = load_image_1(input_image,62892,65535)
# # img = load_image(input_image)
# tic = time.time()
# enhanced = predict_img(model, img)
# print(time.time()-tic)
# save_image(normalize_wafer(enhanced), output_image)
# # save_image(enhanced, output_image)


# print("Saved enhanced image:", output_image)
tic = time.time()
nk = 3
for i in range(0, 4**nk):
    input_image = f"G:\\AutoImageProcessing\\WaferData\\Image1\\Invert_20260323044740\\Branch_{nk}_{i}.tif"
    output_image = f"Result\\output\\Branch_{nk}_{i}.tif"
    img = load_image_1(input_image,63500,65535)
    enhanced = predict_img(model, img)
    save_image(normalize_wafer(enhanced), output_image)
    print("Saved enhanced image:", output_image)
print("Total time:", time.time()-tic)