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
img_1_green = img_1[:,:,1]   # BGR
utils.print_image_details_and_show(img_1_green, "green")

img_1_green_path = os.path.join(output_folder, "ps0-2-b-1.png")
cv.imwrite(img_1_green_path, img_1_green)


# (c)
img_1_red = img_1[:,:,2]   # BGR
utils.print_image_details_and_show(img_1_red, "red")

img_1_red_path = os.path.join(output_folder, "ps0-2-c-1.png")
cv.imwrite(img_1_red_path, img_1_red)


### Q3
# (a)
rows_red, cols_red = img_1_red.shape
rows_center, cols_center = 100, 100
rows_red_l = rows_red//2 -rows_center//2
rows_red_u = rows_red//2 +rows_center//2
cols_red_l = cols_red//2 -cols_center//2
cols_red_u = cols_red//2 +cols_center//2
print(rows_red_l, rows_red_u, cols_red_l, cols_red_u)

img_1_red_center = img_1_red.copy()
img_1_red_center[rows_red_l:rows_red_u, cols_red_l:cols_red_u] = \
    img_1_green[rows_red_l:rows_red_u, cols_red_l:cols_red_u]

utils.print_image_details_and_show(img_1_red_center, "center")

img_1_red_center_path = os.path.join(output_folder, "ps0-3-a-1.png")
cv.imwrite(img_1_red_center_path, img_1_red_center)



















