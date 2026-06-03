import cv2
import numpy as np
input_image = r"G:\AutoImageProcessing\WaferData\FimilarOutput\Slot 5 No Scribe, 11-20, YStep15, Inv, HC\Output\Branch_3_21.tif"
image_temp = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
k = 2
h,w = int(image_temp.shape[0]/k), int(image_temp.shape[1]/k)
for x in range(k):
    for y in range(k):
        output_image = f"G:\\AutoImageProcessing\\Custom-Pytorch-UNet\\Result\\Branch_C{4}_{x*k+y}.tif"
        crop_img = image_temp[x*h:(x+1)*h,y*w:(y+1)*w]
        cv2.imwrite(output_image, crop_img.astype(np.uint16))