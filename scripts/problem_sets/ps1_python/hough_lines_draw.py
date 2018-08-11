import cv2
import numpy as np


def hough_lines_draw(img, peaks_arr, rho_arr, theta_arr, colour_tup=(0,255,0)):
    # Draw lines found in an image using Hough transform.

    # img: Image on top of which to draw lines
    # peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
    # rho: Vector of rho values, in pixels
    # theta: Vector of theta values, in degrees


    img = img.copy()


    ### Initialize
    cos_theta_arr = np.cos(np.deg2rad(theta_arr))  # Radians
    sin_theta_arr = np.sin(np.deg2rad(theta_arr))  # Radians

    def gen_point_on_line(x, y, rho, cos_theta, sin_theta):
        # Takes in x and y pair and generates a point on the line
        # Cases below handle horizontal, vertical, and other lines
        if cos_theta == 0:
            y = rho / sin_theta
        elif sin_theta == 0:
            x = rho / cos_theta
        else:
            y = (rho - x*cos_theta) / sin_theta   # Generate y from x

        x, y = int(round(x)), int(round(y))
        return (x, y)


    ### Draw Lines

    for rho_idx, theta_idx in peaks_arr:
        rho = rho_arr[rho_idx]
        sin_theta = sin_theta_arr[theta_idx]
        cos_theta = cos_theta_arr[theta_idx]

        # Generate points to draw line
        # Points are out of image bounds to make opencv draw an extended line
        x, y = -1, -1
        pt1 = gen_point_on_line(x, y, rho, cos_theta, sin_theta)
        x = y = max(img.shape) +1
        pt2 = gen_point_on_line(x, y, rho, cos_theta, sin_theta)

        img = cv2.line(img, pt1, pt2, colour_tup)   # Draw line from 2 pts


    return img


