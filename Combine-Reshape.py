import cv2
import numpy as np

path_in_folder = r"G:\AutoImageProcessing\Pytorch-UNet\output\\"
step = 8
height = 2048
width = 2048


listimage = []
for i in range(step):
    image = np.array(cv2.imread(path_in_folder+"Branch_3_{}.tif".format(i*step)))
    for j in range(1,step):
        image = np.concatenate((image,np.array(cv2.imread(path_in_folder+"Branch_3_{}.tif".format(i*step+j)))), axis =1)
    listimage.append(image)
    
image_original = listimage[0]
for i in range(1,step):
    image_original = np.concatenate((image_original, listimage[i]))

resized_image = cv2.resize(image_original, (height, width))

cv2.imwrite(r"G:\AutoImageProcessing\Pytorch-UNet\ResizeImage2.tif", resized_image)
# cv2.imwrite(r"G:\AutoImageProcessing\Pytorch-UNet\OriginalImage.tif", image_original)
