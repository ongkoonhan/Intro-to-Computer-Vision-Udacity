import os
import numpy as np
import cv2 as cv2
from shutil import copy2
from time import time

from Intro_To_Computer_Vision_Utils import utils

from hough_lines_acc import hough_lines_acc
from hough_peaks import hough_peaks, mark_hough_peaks
from hough_lines_draw import hough_lines_draw
from hough_circles_acc import hough_circles_acc
from hough_circles_draw import hough_circles_draw
from find_circles import find_circles


input_folder = os.path.join("input")
output_folder = os.path.join("output")

show_img = False


### Q1 ------------------------------------------------------------
# (a)
img_path = os.path.join(input_folder, "ps1-input0.png")
img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

utils.print_image_details_and_show(img, "img", show_img)

img_edges = cv2.Canny(img, 100, 200)

utils.print_image_details_and_show(img_edges, "img_edges", show_img)
img_path = os.path.join(output_folder, "ps1-1-a-1.png")
cv2.imwrite(img_path, img_edges, params=[cv2.IMWRITE_PNG_BILEVEL, 1])   # Save as Binary/Bi-level Image


### Q2 ------------------------------------------------------------
# (a)
img_path = os.path.join(output_folder, "ps1-1-a-1.png")
img_edges = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

hough_space_accum, theta_arr, rho_arr = hough_lines_acc(img_edges)
# hough_space_accum, theta_arr, rho_arr = hough_lines_acc(img_edges, rho_resolution=5)
# hough_space_accum, theta_arr, rho_arr = hough_lines_acc(img_edges, rho_resolution=10)

hough_space_img = hough_space_accum
normalization_factor = 255.0 / np.amax(hough_space_img, axis=None)
hough_space_img = np.uint8(hough_space_img * normalization_factor)   # Convert to unit8

utils.print_image_details_and_show(hough_space_img, "hough_space_img", show_img)
img_path = os.path.join(output_folder, "ps1-2-a-1.png")
cv2.imwrite(img_path, hough_space_img)


# (b)
# Get peaks coordinates
n_peaks = 10
peaks_arr = hough_peaks(hough_space_accum, numpeaks=n_peaks)
# peaks = hough_peaks(hough_space_accum, nHoodSize=(1,1), numpeaks=n_peaks)

print(peaks_arr)
print(len(peaks_arr))

# Draw box around peaks
hough_space_img_marked = hough_space_img.copy()

rect_size = (5,5)
hough_space_img_marked = mark_hough_peaks(hough_space_img_marked, peaks_arr, rect_size)

utils.print_image_details_and_show(hough_space_img_marked, "hough_space_img_marked", show_img)
img_path = os.path.join(output_folder, "ps1-2-b-1.png")
cv2.imwrite(img_path, hough_space_img_marked)


# (c)
img_hough_lines = img.copy()
img_hough_lines = cv2.cvtColor(img_hough_lines, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_lines = hough_lines_draw(img_hough_lines, peaks_arr, rho_arr, theta_arr)

utils.print_image_details_and_show(img_hough_lines, "img_hough_lines", show_img)
img_path = os.path.join(output_folder, "ps1-2-c-1.png")
cv2.imwrite(img_path, img_hough_lines)


### Q3 ------------------------------------------------------------
# (a)
# Gaussian Smoothing
img_path = os.path.join(input_folder, "ps1-input0-noise.png")
img_noise = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
img_noise_smoothed = cv2.GaussianBlur(img_noise, (0,0), 4.0)

utils.print_image_details_and_show(img_noise_smoothed, "img_noise_smoothed", show_img)
img_path = os.path.join(output_folder, "ps1-3-a-1.png")
cv2.imwrite(img_path, img_noise_smoothed)


# Canny edge detection
img_noise_edges = cv2.Canny(img_noise, 900, 1100)

utils.print_image_details_and_show(img_noise_edges, "img_noise_edges", show_img)
img_path = os.path.join(output_folder, "ps1-3-b-1.png")
cv2.imwrite(img_path, img_noise_edges, params=[cv2.IMWRITE_PNG_BILEVEL, 1])

img_noise_smoothed_edges = cv2.Canny(img_noise_smoothed, 40, 60)

utils.print_image_details_and_show(img_noise_smoothed_edges, "img_noise_smoothed_edges", show_img)
img_path = os.path.join(output_folder, "ps1-3-b-2.png")
cv2.imwrite(img_path, img_noise_smoothed_edges, params=[cv2.IMWRITE_PNG_BILEVEL, 1])


# Hough Transform
hough_space_accum, theta_arr, rho_arr = hough_lines_acc(img_noise_smoothed_edges, rho_resolution=0.5)

hough_space_img = hough_space_accum
normalization_factor = 255.0 / np.amax(hough_space_img, axis=None)
hough_space_img = np.uint8(hough_space_img * normalization_factor)   # Convert to unit8

utils.print_image_details_and_show(hough_space_img, "hough_space_img", show_img)

n_peaks = 10
peaks_arr = hough_peaks(hough_space_accum, numpeaks=n_peaks)

print(peaks_arr)
print(len(peaks_arr))


# Draw box around peaks
hough_space_img_marked = hough_space_img.copy()

rect_size = (5,5)
hough_space_img_marked = mark_hough_peaks(hough_space_img_marked, peaks_arr, rect_size)

utils.print_image_details_and_show(hough_space_img_marked, "hough_space_img_marked", show_img)
img_path = os.path.join(output_folder, "ps1-3-c-1.png")
cv2.imwrite(img_path, hough_space_img_marked)


# Draw Lines
img_hough_lines = img_noise_smoothed.copy()
img_hough_lines = cv2.cvtColor(img_hough_lines, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_lines = hough_lines_draw(img_hough_lines, peaks_arr, rho_arr, theta_arr)

utils.print_image_details_and_show(img_hough_lines, "img_hough_lines", show_img)
img_path = os.path.join(output_folder, "ps1-3-c-2.png")
cv2.imwrite(img_path, img_hough_lines)


### Q4 ------------------------------------------------------------
# (a)
# Gaussian Smoothing
img_path = os.path.join(input_folder, "ps1-input1.png")
img1 = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
img1_smoothed = cv2.GaussianBlur(img1, (0,0), 2.0)

utils.print_image_details_and_show(img1_smoothed, "img1_smoothed", show_img)
img_path = os.path.join(output_folder, "ps1-4-a-1.png")
cv2.imwrite(img_path, img1_smoothed)


# Canny edge detection
img1_smoothed_edges = cv2.Canny(img1_smoothed, 100, 200)

utils.print_image_details_and_show(img1_smoothed_edges, "img1_smoothed_edges", show_img)
img_path = os.path.join(output_folder, "ps1-4-b-1.png")
cv2.imwrite(img_path, img1_smoothed_edges, params=[cv2.IMWRITE_PNG_BILEVEL, 1])


# Hough Transform
hough_space_accum, theta_arr, rho_arr = hough_lines_acc(img1_smoothed_edges, rho_resolution=1)

hough_space_img = hough_space_accum
normalization_factor = 255.0 / np.amax(hough_space_img, axis=None)
hough_space_img = np.uint8(hough_space_img * normalization_factor)   # Convert to unit8

utils.print_image_details_and_show(hough_space_img, "hough_space_img", show_img)

n_peaks = 10
peaks_arr = hough_peaks(hough_space_accum, numpeaks=n_peaks)

print(peaks_arr)
print(len(peaks_arr))


# Draw box around peaks
hough_space_img_marked = hough_space_img.copy()

rect_size = (5,5)
hough_space_img_marked = mark_hough_peaks(hough_space_img_marked, peaks_arr, rect_size)

utils.print_image_details_and_show(hough_space_img_marked, "hough_space_img_marked", show_img)
img_path = os.path.join(output_folder, "ps1-4-c-1.png")
cv2.imwrite(img_path, hough_space_img_marked)


# Draw Lines
img_hough_lines = img1.copy()
img_hough_lines = cv2.cvtColor(img_hough_lines, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_lines = hough_lines_draw(img_hough_lines, peaks_arr, rho_arr, theta_arr)

utils.print_image_details_and_show(img_hough_lines, "img_hough_lines", show_img)
img_path = os.path.join(output_folder, "ps1-4-c-2.png")
cv2.imwrite(img_path, img_hough_lines)


### Q5 ------------------------------------------------------------
# (a)
copy2(os.path.join(output_folder, "ps1-4-a-1.png"), os.path.join(output_folder, "ps1-5-a-1.png"))
copy2(os.path.join(output_folder, "ps1-4-b-1.png"), os.path.join(output_folder, "ps1-5-a-2.png"))

img_path = os.path.join(output_folder, "ps1-5-a-2.png")
img2_edges = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

utils.print_image_details_and_show(img2_edges, "img2_edges", show_img)


# Circle Hough Transform
radius = 20
hough_space_accum = hough_circles_acc(img2_edges, radius)

hough_space_img = hough_space_accum
normalization_factor = 255.0 / np.amax(hough_space_img, axis=None)
hough_space_img = np.uint8(hough_space_img * normalization_factor)   # Convert to unit8

utils.print_image_details_and_show(hough_space_img, "hough_space_img", show_img)

n_peaks = 10
peaks_arr = hough_peaks(hough_space_accum, numpeaks=n_peaks)

print(peaks_arr)
print(len(peaks_arr))


# Draw box around peaks
hough_space_img_marked = hough_space_img.copy()

rect_size = (7,7)
hough_space_img_marked = mark_hough_peaks(hough_space_img_marked, peaks_arr, rect_size)

utils.print_image_details_and_show(hough_space_img_marked, "hough_space_img_marked", show_img)
img_path = os.path.join(output_folder, "ps1-5-a-2-misc.png")
cv2.imwrite(img_path, hough_space_img_marked)


# Draw Lines
img_path = os.path.join(input_folder, "ps1-input1.png")
img_hough_circles = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

img_hough_circles = cv2.cvtColor(img_hough_circles, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_circles = hough_circles_draw(img_hough_circles, peaks_arr, radius, "numeric")

utils.print_image_details_and_show(img_hough_circles, "img_hough_circles", show_img)
img_path = os.path.join(output_folder, "ps1-5-a-3.png")
cv2.imwrite(img_path, img_hough_circles)


# (b)
img_path = os.path.join(output_folder, "ps1-5-a-2.png")
img2_edges = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

utils.print_image_details_and_show(img2_edges, "img2_edges", show_img)


# Circle Hough Transform
t0 = time()

radius_range = (20,35)
n_peaks = 20
radius_step = 2
centers_arr, radii_arr = find_circles(img2_edges, radius_range, n_peaks, radius_step=radius_step)

print("time taken for Hough Circle is {:.3f} seconds".format(time()-t0))
print(centers_arr)
print(len(centers_arr))


# # Draw Lines
img_path = os.path.join(input_folder, "ps1-input1.png")
img_hough_circles = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

img_hough_circles = cv2.cvtColor(img_hough_circles, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_circles = hough_circles_draw(img_hough_circles, centers_arr, radii_arr, "array")

utils.print_image_details_and_show(img_hough_circles, "img_hough_circles", show_img)
img_path = os.path.join(output_folder, "ps1-5-b-1.png")
cv2.imwrite(img_path, img_hough_circles)


### Q6 ------------------------------------------------------------
# (a)
# Gaussian Smoothing
img_path = os.path.join(input_folder, "ps1-input2.png")
img3 = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
img3_smoothed = cv2.GaussianBlur(img3, (0,0), 1.0)

utils.print_image_details_and_show(img3_smoothed, "img1_smoothed", show_img)


# Canny edge detection
img3_smoothed_edges = cv2.Canny(img3_smoothed, 100, 200)

utils.print_image_details_and_show(img3_smoothed_edges, "img1_smoothed_edges", show_img)


# Hough Transform
hough_space_accum, theta_arr, rho_arr = hough_lines_acc(img3_smoothed_edges, rho_resolution=1)

n_peaks = 10
peaks_arr = hough_peaks(hough_space_accum, numpeaks=n_peaks)

print(peaks_arr)
print(len(peaks_arr))


# Draw Lines
img_hough_lines = img3_smoothed.copy()
img_hough_lines = cv2.cvtColor(img_hough_lines, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_lines = hough_lines_draw(img_hough_lines, peaks_arr, rho_arr, theta_arr)

utils.print_image_details_and_show(img_hough_lines, "img_hough_lines", show_img)
img_path = os.path.join(output_folder, "ps1-6-a-1.png")
cv2.imwrite(img_path, img_hough_lines)


# (c)
# Keep parallel lines that are near enough
rho_threshold = 50
theta_threshold = 1

parallel_lines_list = []
for i in range(len(peaks_arr)):
    # Create thresholds for rho and theta
    rho, theta = peaks_arr[i]
    rho_l, rho_u = rho - rho_threshold, rho + rho_threshold
    theta_l, theta_u = theta - theta_threshold, theta + theta_threshold
    for j in range(i +1, len(peaks_arr), 1):
        rho2, theta2 = peaks_arr[j]
        if theta_l <= theta2 and theta2 <= theta_u:   # Within theta tolerance (parallel lines)
            if rho_l <= rho2 and rho2 <= rho_u:   # Outside rho threshold (far away lines)
                parallel_lines_list.append(i)
                parallel_lines_list.append(j)   # Add j index to list

parallel_lines_set = set(parallel_lines_list)
peaks_arr = peaks_arr[np.asarray(tuple(parallel_lines_set))]

print(peaks_arr)
print(len(peaks_arr))


# Draw Lines
img_hough_lines = img3_smoothed.copy()
img_hough_lines = cv2.cvtColor(img_hough_lines, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_lines = hough_lines_draw(img_hough_lines, peaks_arr, rho_arr, theta_arr)

utils.print_image_details_and_show(img_hough_lines, "img_hough_lines", show_img)
img_path = os.path.join(output_folder, "ps1-6-c-1.png")
cv2.imwrite(img_path, img_hough_lines)


### Q7 ------------------------------------------------------------
# (a)
# Gaussian Smoothing
img_path = os.path.join(input_folder, "ps1-input2.png")
img3 = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

img_path = os.path.join(output_folder, "ps1-7-a-1_misc0_original_image.png")
cv2.imwrite(img_path, img3)

img3 = cv2.subtract(cv2.multiply(img3, 1.5), 50)   # Contrast x1.5 and darken by 50
# img3_smoothed = cv2.GaussianBlur(img3, (0,0), 1.1)
img3_smoothed = cv2.GaussianBlur(img3, (0,0), 1.2)

img_path = os.path.join(output_folder, "ps1-7-a-1_misc1_high_contrast_smoothed.png")
cv2.imwrite(img_path, img3_smoothed)


# Canny edge detection
img3_smoothed_edges = cv2.Canny(img3_smoothed, 100, 300)

img_path = os.path.join(output_folder, "ps1-7-a-1_misc2_edges_with_blobs.png")
cv2.imwrite(img_path, img3_smoothed_edges)


# Remove "blobs" by subtracting a smoothed version of the edge image
img3_smoothed_edges_blured = cv2.GaussianBlur(img3_smoothed_edges, (0,0), 7.0)
# normalization_factor = 255.0 / np.amax(img3_smoothed_edges_blured, axis=None)
normalization_factor = 255.0 / np.percentile(img3_smoothed_edges_blured, 99.0, axis=None)
img3_smoothed_edges_blured = cv2.multiply(img3_smoothed_edges_blured, normalization_factor)

img_path = os.path.join(output_folder, "ps1-7-a-1_misc3_edges_blured.png")
cv2.imwrite(img_path, img3_smoothed_edges_blured)


for i in range(5):   # Subtraction and thresholding
    img3_smoothed_edges = cv2.subtract(img3_smoothed_edges, img3_smoothed_edges_blured)
    # img3_smoothed_edges[img3_smoothed_edges > 100] = 255   # Keep stronger edges (non-blobs)
    img3_smoothed_edges[img3_smoothed_edges > 50] = 255   # Keep stronger edges (non-blobs)

img3_smoothed_edges[img3_smoothed_edges > 0] = 255

img_path = os.path.join(output_folder, "ps1-7-a-1_misc4_edges_no_blobs.png")
cv2.imwrite(img_path, img3_smoothed_edges)

cv2.destroyAllWindows()


# Circle Hough Transform
t0 = time()

radius_range = (25,35)
n_peaks = 30
radius_step = 1
centers_arr, radii_arr = find_circles(img3_smoothed_edges, radius_range, n_peaks, radius_step=radius_step)

print("time taken for Hough Circle is {:.3f} seconds".format(time()-t0))
print(centers_arr)
print(len(centers_arr))


# # Draw Lines
img_path = os.path.join(input_folder, "ps1-input2.png")
img_hough_circles = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
img_hough_circles = cv2.cvtColor(img_hough_circles, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_circles = hough_circles_draw(img_hough_circles, centers_arr, radii_arr, "array")

utils.print_image_details_and_show(img_hough_circles, "img_hough_circles", show_img)
img_path = os.path.join(output_folder, "ps1-7-a-1.png")
cv2.imwrite(img_path, img_hough_circles)



### Q8 ------------------------------------------------------------
# (a)
# LINE DETECTION
# Gaussian Smoothing
img_path = os.path.join(input_folder, "ps1-input3.png")
img_orig = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
img_high_contrast = cv2.subtract(cv2.multiply(img_orig, 1.8), 50)   # Contrast x1.8 and darken by 100
img_smoothed = cv2.GaussianBlur(img_high_contrast, (0,0), 2.0)

utils.print_image_details_and_show(img_smoothed, "img_smoothed", show_img)


# Canny edge detection
img_smoothed_edges = cv2.Canny(img_smoothed, 110, 300)

utils.print_image_details_and_show(img_smoothed_edges, "img_smoothed_edges", show_img)


# Hough Transform
hough_space_accum, theta_arr, rho_arr = hough_lines_acc(img_smoothed_edges, rho_resolution=1)

n_peaks = 30
threshold = 0.30 * np.amax(hough_space_accum, axis=None)
peaks_arr = hough_peaks(hough_space_accum, numpeaks=n_peaks, threshold=threshold)

print(peaks_arr)
print(len(peaks_arr))


# Draw Lines
img_hough_lines = img_orig.copy()
img_hough_lines = cv2.cvtColor(img_hough_lines, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_lines = hough_lines_draw(img_hough_lines, peaks_arr, rho_arr, theta_arr)

utils.print_image_details_and_show(img_hough_lines, "img_hough_lines", show_img)


# Keep parallel lines that are near enough
rho_threshold = 20
theta_threshold = 1

parallel_lines_list = []
for i in range(len(peaks_arr)):
    # Create thresholds for rho and theta
    rho, theta = peaks_arr[i]
    rho_l, rho_u = rho - rho_threshold, rho + rho_threshold
    theta_l, theta_u = theta - theta_threshold, theta + theta_threshold
    for j in range(i +1, len(peaks_arr), 1):
        rho2, theta2 = peaks_arr[j]
        if theta_l <= theta2 and theta2 <= theta_u:   # Within theta tolerance (parallel lines)
            if rho_l <= rho2 and rho2 <= rho_u:   # Outside rho threshold (far away lines)
                parallel_lines_list.append(i)
                parallel_lines_list.append(j)   # Add j index to list

parallel_lines_set = set(parallel_lines_list)
peaks_arr = peaks_arr[np.asarray(tuple(parallel_lines_set))]

print("After filtering close parallel lines:")
print(peaks_arr)
print(len(peaks_arr))


# Draw Lines
img_hough_lines = img_orig.copy()
img_hough_lines = cv2.cvtColor(img_hough_lines, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_lines = hough_lines_draw(img_hough_lines, peaks_arr, rho_arr, theta_arr)

utils.print_image_details_and_show(img_hough_lines, "img_hough_lines", show_img)


# CIRCLE DETECTION
# Gaussian Smoothing
img_high_contrast = cv2.subtract(cv2.multiply(img_orig, 2.0), 50)   # Contrast x2 and darken by 50
# img3_smoothed = cv2.GaussianBlur(img3, (0,0), 1.1)
img_smoothed = cv2.GaussianBlur(img_high_contrast, (0,0), 1.0)

utils.print_image_details_and_show(img_smoothed, "img_smoothed", show_img)


# Canny edge detection
img_smoothed_edges = cv2.Canny(img_smoothed, 100, 300)

utils.print_image_details_and_show(img_smoothed_edges, "img_smoothed_edges", show_img)


# Remove "blobs" by subtracting a smoothed version of the edge image
img_smoothed_edges_blured = cv2.GaussianBlur(img_smoothed_edges, (0,0), 10.0)
# normalization_factor = 255.0 / np.amax(img_smoothed_edges_blured, axis=None)
normalization_factor = 255.0 / np.percentile(img_smoothed_edges_blured, 99.0, axis=None)
img_smoothed_edges_blured = cv2.multiply(img_smoothed_edges_blured, normalization_factor)

utils.print_image_details_and_show(img_smoothed_edges_blured, "img_smoothed_edges_blured", show_img)

for i in range(5):   # Subtraction and thresholding
    img_smoothed_edges = cv2.subtract(img_smoothed_edges, img_smoothed_edges_blured)
    # img3_smoothed_edges[img3_smoothed_edges > 100] = 255   # Keep stronger edges (non-blobs)
    img_smoothed_edges[img_smoothed_edges > 50] = 255   # Keep stronger edges (non-blobs)

img_smoothed_edges[img_smoothed_edges > 0] = 255

utils.print_image_details_and_show(img_smoothed_edges, "img_smoothed_edges", show_img)

# cv2.destroyAllWindows()


# Circle Hough Transform
t0 = time()

radius_range = (20,35)
n_peaks = 40
radius_step = 5

nHoodSize = img_orig.shape
new_tup_list = [(int(elem / 10) * 2) + 1 for elem in nHoodSize]  # b,a space, create odd values >= size(H) / 10
nHoodSize_radius = int(5 / radius_step)
nHoodSize_radius = nHoodSize_radius + 1 if nHoodSize_radius % 2 == 0 else nHoodSize_radius
new_tup_list.insert(0, nHoodSize_radius)  # Set nHoodSize for radius parameter to be 3
nHoodSize = tuple(new_tup_list)

centers_arr, radii_arr = find_circles(img_smoothed_edges, radius_range, n_peaks, radius_step=radius_step, nHoodSize=nHoodSize)

print("time taken for Hough Circle is {:.3f} seconds".format(time()-t0))
print(centers_arr)
print(len(centers_arr))


# Draw Lines
img_hough_circles = img_hough_lines.copy()
# img_hough_circles = cv2.cvtColor(img_hough_circles, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_circles = hough_circles_draw(img_hough_circles, centers_arr, radii_arr, "array")

utils.print_image_details_and_show(img_hough_circles, "img_hough_circles", show_img)

img_path = os.path.join(output_folder, "ps1-8-a-1.png")
cv2.imwrite(img_path, img_hough_circles)


# (c)
# Choose only significant overlapping circles
sorted_radii_idx = np.flip(np.argsort(radii_arr), 0)   # Sort circles from largest to smallest (returns sorted index)
len_idx = len(sorted_radii_idx)

new_centres_list, new_radii_list = [], []
for i in range(len_idx):
    if sorted_radii_idx[i] == -1:
        continue

    big_c_idx = sorted_radii_idx[i]
    big_c_rad, big_c_cen = radii_arr[big_c_idx], centers_arr[big_c_idx]

    circle_combine_list = [big_c_idx]
    for j in range(i +1, len_idx, 1):
        small_c_idx = sorted_radii_idx[j]
        small_c_rad, small_c_cen = radii_arr[small_c_idx], centers_arr[small_c_idx]
        centers_dist = np.linalg.norm(np.subtract(big_c_cen, small_c_cen))

        # Check if more than half of the smaller circle is in the bigger circle
        if centers_dist < big_c_rad:
            circle_combine_list.append(small_c_idx)
            sorted_radii_idx[j] = -1   # Smaller circle will not be checked again

    # Combine overlapping circles by averaging
    if len(circle_combine_list) == 1:
        continue

    # circle_combine_idx_arr = np.asarray(tuple(circle_combine_list))
    circle_combine_idx_arr = np.array(circle_combine_list)
    new_c_rad = int(np.mean(radii_arr[circle_combine_idx_arr]))
    new_c_cen = np.uint16(np.mean(centers_arr[circle_combine_idx_arr], axis=0))

    new_radii_list.append(new_c_rad)
    new_centres_list.append(new_c_cen)

centers_arr, radii_arr = np.array(new_centres_list), np.array(new_radii_list)

print("After combining circles:")
print(centers_arr)
print(len(centers_arr))


# Draw Lines
img_hough_circles = img_hough_lines.copy()
# img_hough_circles = cv2.cvtColor(img_hough_circles, cv2.COLOR_GRAY2BGR)   # Convert to colour to draw coloured line
img_hough_circles = hough_circles_draw(img_hough_circles, centers_arr, radii_arr, "array")

utils.print_image_details_and_show(img_hough_circles, "img_hough_circles", show_img)

img_path = os.path.join(output_folder, "ps1-8-c-1.png")
cv2.imwrite(img_path, img_hough_circles)


































