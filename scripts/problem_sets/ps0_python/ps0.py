import os
import numpy as np
import cv2 as cv

from Intro_To_Computer_Vision_Utils import utils


output_folder = os.path.join("output")


### Q1
# (a)
img_1_path = os.path.join(output_folder, 'ps0-1-a-1.png')
img_2_path = os.path.join(output_folder, 'ps0-1-a-2.png')
img_1 = cv.imread(img_1_path, flags=cv.IMREAD_UNCHANGED)
img_2 = cv.imread(img_2_path, flags=cv.IMREAD_UNCHANGED)

# utils.print_image_details_and_show(img_1)
# utils.print_image_details_and_show(img_2)


### Q2
# (a)
utils.print_image_details_and_show(img_1, "before swap")

swap_index = np.array([2,1,0])   # Indices to change BGR to RGB
img_1_swap = img_1[:,:,swap_index]

utils.print_image_details_and_show(img_1_swap, "after swap")

img_1_swap_path = os.path.join(output_folder, "ps0-2-a-1.png")
cv.imwrite(img_1_swap_path, img_1_swap)


# (b)
img_1_green = img_1[:,:,1]
img_1_green_path = os.path.join(output_folder, "ps0-2-b-1.png")
cv.imwrite(img_1_green_path, img_1_green)

























