import os
import numpy as np
import cv2 as cv

from Intro_To_Computer_Vision_Utils import utils


output_folder = os.path.join("output")


show_img = False


### Q1
# (a)
img_1_path = os.path.join(output_folder, 'ps0-1-a-1.png')
img_2_path = os.path.join(output_folder, 'ps0-1-a-2.png')
img_1 = cv.imread(img_1_path, flags=cv.IMREAD_UNCHANGED)
img_2 = cv.imread(img_2_path, flags=cv.IMREAD_UNCHANGED)

# utils.print_image_details_and_show(img_1, show_img)
# utils.print_image_details_and_show(img_2, show_img)


### Q2
# (a)
utils.print_image_details_and_show(img_1, "before swap", show_img)

swap_index = np.array([2,1,0])   # Indices to change BGR to RGB
img_1_swap = img_1[:,:,swap_index]

utils.print_image_details_and_show(img_1_swap, "after swap", show_img)

img_path = os.path.join(output_folder, "ps0-2-a-1.png")
cv.imwrite(img_path, img_1_swap)


# (b)
img_1_green = img_1[:,:,1]   # BGR
utils.print_image_details_and_show(img_1_green, "green", show_img)

img_path = os.path.join(output_folder, "ps0-2-b-1.png")
cv.imwrite(img_path, img_1_green)


# (c)
img_1_red = img_1[:,:,2]   # BGR
utils.print_image_details_and_show(img_1_red, "red", show_img)

img_path = os.path.join(output_folder, "ps0-2-c-1.png")
cv.imwrite(img_path, img_1_red)


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

utils.print_image_details_and_show(img_1_red_center, "center", show_img)

img_path = os.path.join(output_folder, "ps0-3-a-1.png")
cv.imwrite(img_path, img_1_red_center)


### Q4
# (a)
min_pix_val, max_pix_val, minLoc, maxLoc	= cv.minMaxLoc(img_1_green)
mean_pix_val, sd_pix_val = cv.meanStdDev(img_1_green)

print("max: {0}, min: {1}, mean: {2}, SD: {3}" \
      .format(max_pix_val, min_pix_val, mean_pix_val, sd_pix_val)
)


# (b)
img_1_green_normalized = img_1_green.copy()
img_1_green_normalized = cv.subtract(img_1_green_normalized, mean_pix_val)
img_1_green_normalized = cv.divide(img_1_green_normalized, sd_pix_val)
img_1_green_normalized = cv.multiply(img_1_green_normalized, 10)
img_1_green_normalized = cv.add(img_1_green_normalized, mean_pix_val)

utils.print_image_details_and_show(img_1_green_normalized, "green_normalized", show_img)

img_path = os.path.join(output_folder, "ps0-4-b-1.png")
cv.imwrite(img_path, img_1_green_normalized)


# (c)
rows, cols = img_1_green.shape

# Translate in x direction by -2 (shift left)
translation_matx = np.float32([ [1,0,-2],
                                [0,1,0] ])
img_1_green_left = img_1_green.copy()
img_1_green_left = cv.warpAffine(img_1_green_left, translation_matx, (cols,rows))

utils.print_image_details_and_show(img_1_green_left, "green_shifted_left", show_img)

img_path = os.path.join(output_folder, "ps0-4-c-1.png")
cv.imwrite(img_path, img_1_green_left)


# (d)
img_1_green_diff = cv.subtract(img_1_green, img_1_green_left)

utils.print_image_details_and_show(img_1_green_diff, "green_difference", show_img)

img_path = os.path.join(output_folder, "ps0-4-d-1.png")
cv.imwrite(img_path, img_1_green_diff)



























