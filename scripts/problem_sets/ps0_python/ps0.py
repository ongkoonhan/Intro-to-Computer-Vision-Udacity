import os
import numpy as np
import cv2 as cv2

from Intro_To_Computer_Vision_Utils import utils


output_folder = os.path.join("output")


show_img = False


### Q1
# (a)
img_1_path = os.path.join(output_folder, 'ps0-1-a-1.png')
img_2_path = os.path.join(output_folder, 'ps0-1-a-2.png')
img_1 = cv2.imread(img_1_path, flags=cv2.IMREAD_UNCHANGED)
img_2 = cv2.imread(img_2_path, flags=cv2.IMREAD_UNCHANGED)

# utils.print_image_details_and_show(img_1, show_img)
# utils.print_image_details_and_show(img_2, show_img)


### Q2
# (a)
utils.print_image_details_and_show(img_1, "before swap", show_img)

swap_index = np.array([2,1,0])   # Indices to change BGR to RGB
img_1_swap = img_1[:,:,swap_index]

utils.print_image_details_and_show(img_1_swap, "after swap", show_img)

img_path = os.path.join(output_folder, "ps0-2-a-1.png")
cv2.imwrite(img_path, img_1_swap)


# (b)
img_1_green = img_1[:,:,1]   # BGR
utils.print_image_details_and_show(img_1_green, "green", show_img)

img_path = os.path.join(output_folder, "ps0-2-b-1.png")
cv2.imwrite(img_path, img_1_green)


# (c)
img_1_red = img_1[:,:,2]   # BGR
utils.print_image_details_and_show(img_1_red, "red", show_img)

img_path = os.path.join(output_folder, "ps0-2-c-1.png")
cv2.imwrite(img_path, img_1_red)


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
cv2.imwrite(img_path, img_1_red_center)


### Q4
# (a)
min_pix_val, max_pix_val, minLoc, maxLoc	= cv2.minMaxLoc(img_1_green)
mean_pix_val, sd_pix_val = cv2.meanStdDev(img_1_green)

print("max: {0}, min: {1}, mean: {2}, SD: {3}" \
      .format(max_pix_val, min_pix_val, mean_pix_val, sd_pix_val)
)


# (b)
img_1_green_normalized = img_1_green.copy()
img_1_green_normalized = cv2.subtract(img_1_green_normalized, mean_pix_val)
img_1_green_normalized = cv2.divide(img_1_green_normalized, sd_pix_val)
img_1_green_normalized = cv2.multiply(img_1_green_normalized, 10)
img_1_green_normalized = cv2.add(img_1_green_normalized, mean_pix_val)

utils.print_image_details_and_show(img_1_green_normalized, "green_normalized", show_img)

img_path = os.path.join(output_folder, "ps0-4-b-1.png")
cv2.imwrite(img_path, img_1_green_normalized)


# (c)
rows, cols = img_1_green.shape

# Translate in x direction by -2 (shift left)
# Translation matrix (augmented matrix)
translation_matx = np.float32([ [1,0,-2],
                                [0,1,0] ])
img_1_green_left = img_1_green.copy()
img_1_green_left = cv2.warpAffine(img_1_green_left, translation_matx, (cols,rows))

utils.print_image_details_and_show(img_1_green_left, "green_shifted_left", show_img)

img_path = os.path.join(output_folder, "ps0-4-c-1.png")
cv2.imwrite(img_path, img_1_green_left)


# (d)
img_1_green_diff = cv2.subtract(img_1_green, img_1_green_left)

utils.print_image_details_and_show(img_1_green_diff, "green_difference", show_img)

img_path = os.path.join(output_folder, "ps0-4-d-1.png")
cv2.imwrite(img_path, img_1_green_diff)


# Q5
# (a)
show_img_loop = False
for stddev in range(10,101,10):
    img_1_green_noise_gauss = img_1_green.copy()
    noise_gauss_green = np.zeros(shape=img_1_green_noise_gauss.shape, dtype=np.uint8)
    noise_gauss_green = cv2.randn(noise_gauss_green, mean=0, stddev=stddev)

    img_1_green_noise_gauss = cv2.add(img_1_green_noise_gauss, noise_gauss_green)
    img_1_noise_green = img_1.copy()
    img_1_noise_green[:,:,1] = img_1_green_noise_gauss

    utils.print_image_details_and_show(img_1_noise_green, "green_noise, stddev: {0}".format(stddev), show_img_loop)


selected_stddev = 50
img_1_green_noise_gauss = img_1_green.copy()
noise_gauss_green = np.zeros(shape=img_1_green_noise_gauss.shape, dtype=np.uint8)
noise_gauss_green = cv2.randn(noise_gauss_green, mean=0, stddev=selected_stddev)

img_1_green_noise_gauss = cv2.add(img_1_green_noise_gauss, noise_gauss_green)
img_1_noise_green = img_1.copy()
img_1_noise_green[:,:,1] = img_1_green_noise_gauss

utils.print_image_details_and_show(img_1_noise_green, "green_noise, stddev: {0}".format(selected_stddev), show_img)

img_path = os.path.join(output_folder, "ps0-5-a-1.png")
cv2.imwrite(img_path, img_1_noise_green)


# (b)
img_1_blue_noise_gauss = img_1[:,:,0].copy()   # BGR
noise_gauss_blue = noise_gauss_green.copy()

img_1_blue_noise_gauss = cv2.add(img_1_blue_noise_gauss, noise_gauss_blue)
img_1_noise_blue = img_1.copy()
img_1_noise_blue[:,:,0] = img_1_blue_noise_gauss

utils.print_image_details_and_show(img_1_noise_green, "green_noise, stddev: {0}".format(selected_stddev), show_img)

img_path = os.path.join(output_folder, "ps0-5-b-1.png")
cv2.imwrite(img_path, img_1_noise_blue)
















