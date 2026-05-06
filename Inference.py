import torch
import numpy as np
import cv2
from pathlib import Path
import time
from unet import UNet   # model của bạn
from unet.student_unet_model import StudentUNet,StudentUNet2,StudentUNet3

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

def load_image_2(img_path,p1,p99):

    # img = cv2.imread(img_path)[:,:,0]/255.0
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)/65535

    if img is None:
        raise RuntimeError("Cannot load image")

    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    img = normalize_wafer_1(img,p1/65535,p99/65535)

    # img = torch.from_numpy(img).float().unsqueeze(0)

    return img

def predict_img(model, img_tensor):

    img_tensor = img_tensor.to(device)

    with torch.no_grad():

        residual = model(img_tensor)
        
        # enhanced = img_tensor + residual
        enhanced = img_tensor*(1 + residual)
        # print(enhanced)

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
# model_path = "savecheckpoints/checkpoints20260504_32/checkpoint_epoch20.pth"
# model_path = "student_best.pth"

model = UNet(n_channels=1, n_classes=1)
# model = UNet(n_channels=1, n_classes=1, base_c=32)

# model = StudentUNet(
#         n_channels=1,
#         n_classes=1,
#         base_c=16,
#         bilinear=True,
#         return_features=False
#     ).to(device)
# model = StudentUNet3(
#         n_channels=1,
#         n_classes=1,
#         base_c=16,
#         bilinear=False,
#         return_features=False
#     ).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
print(next(model.parameters()).dtype)
model = model.half()
model.to(device)
model.eval()

# input_image = r"G:\AutoImageProcessing\Slot0113.tif"

# # input_image = r"G:\AutoImageProcessing\WaferData\Image2\Invert_20260323053237\Branch_2_4.tif"
# output_image = "enhanced.tif"
# # img = torch.from_numpy(np.load(input_image)).float().unsqueeze(0)#
# # img = load_image_1(input_image,29135,65535)
# img = load_image_1(input_image,54000,60000)
# # img = load_image_1(input_image,62892,65535)
# # img = load_image(input_image)
# print(img.shape)
# for ks in range(300,6201,200):
#     print("kernel_size: ",ks)
#     if ks < img.shape[2]:
#         imtemp = img[:,:,:ks,:ks]
#     else:
#         imtemp = img
#     tic = time.time()
#     enhanced = predict_img(model, imtemp)
#     print("Time: ", time.time()-tic)
# save_image(normalize_wafer(enhanced), output_image)
# save_image(enhanced, output_image)


# print("Saved enhanced image:", output_image)



tic = time.time()
nk = 2
step = 1
for i in range(0, 4**nk, step):
    list_img = []
    for j in range(i, i+step):
        # input_image = f"G:\\AutoImageProcessing\\WaferData\\Image1\\Invert_20260323044740\\Branch_{nk}_{i}.tif"
        # input_image = f"G:\\AutoImageProcessing\\WaferData\\Image2_1\\LocalNormalize_20260423030053\\Branch_{nk}_{j}.tif"
        input_image = f"G:\\AutoImageProcessing\\WaferData\\Image2\\Invert_20260323053237\\Branch_{nk}_{j}.tif"
        # input_image = f"G:\\AutoImageProcessing\\WaferData\\Image2\\Invert-lv4\\Branch_{nk}_{j}.tif"

        # image_temp = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        # h,w = int(image_temp.shape[0]/4), int(image_temp.shape[1]/4)
        # for x in range(4):
        #     for y in range(4):
        #         output_image = f"G:\\AutoImageProcessing\\WaferData\\Image2\\Invert-lv4\\Branch_{4}_{16*j+x*4+y}.tif"
        #         crop_img = image_temp[x*h:(x+1)*h,y*w:(y+1)*w]
        #         cv2.imwrite(output_image, crop_img.astype(np.uint16))

        # list_img.append(load_image(input_image).squeeze(0))
        list_img.append(load_image_2(input_image,29135,65535))
    raw_img = np.array(list_img)
    torch_img= torch.from_numpy(raw_img).float().half()
    enhanced = predict_img(model, torch_img)
    for j in range(i, i+step):
        output_image = f"Result\\output\\Branch_{nk}_{j}.tif"
        save_image(normalize_wafer(enhanced), output_image)#step=1
        # save_image(normalize_wafer(enhanced[j-i]), output_image)#step>1
        print("Saved enhanced image:", output_image)
print("Total time:", time.time()-tic)