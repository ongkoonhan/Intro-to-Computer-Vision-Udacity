import numpy as np


def hough_circles_acc(BW, radius):
    # Compute Hough accumulator array for finding circles.
    #
    # BW: Binary (black and white) image containing edge pixels
    # radius: Radius of circles to look for, in pixels


    img = BW


    ### Checks

    # Binary image check
    assert len(img.shape) == 2, \
        "Input image must be a binary image"


    ### Initialization

    theta_arr = np.deg2rad(np.arange(0, 360, step=1))   # Radians
    cos_theta_arr = np.cos(theta_arr)
    sin_theta_arr = np.sin(theta_arr)

    edge_pts_y_arr, edge_pts_x_arr = np.nonzero(img)   # Indices of points in edge image
    hough_space_accum = np.zeros((img.shape[0], img.shape[1],), dtype=np.uint64)   # b,a space has the same size as y,x space


    ### Run voting procedure in Hough Space

    # We loop through edge pixels and run the voting for each (b,a) pair

    for x, y in zip(edge_pts_x_arr, edge_pts_y_arr):   # Loop through edge pixels
        a_arr = x - radius*cos_theta_arr
        b_arr = y - radius*sin_theta_arr

        a_arr = np.uint16(np.rint(a_arr))   # Round a,b values for hough accum indexing
        b_arr = np.uint16(np.rint(b_arr))

        for a_idx, b_idx in zip(a_arr, b_arr):
            # Negative values from parameter calculation causes Index Out of Bounds Error
            try:
                hough_space_accum[b_idx, a_idx] += 1  # Vote
            except IndexError:
                continue


    return hough_space_accum




























