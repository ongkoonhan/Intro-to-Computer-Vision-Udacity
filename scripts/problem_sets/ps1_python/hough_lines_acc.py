import numpy as np


def hough_lines_acc(BW, rho_resolution=1, theta_range=(-90,89), theta_step=1.0):
    # Compute Hough accumulator array for finding lines.

    # BW: Binary (black and white) image containing edge pixels
    # RhoResolution (optional): Difference between successive rho values, in pixels
    # Theta (optional): Vector of theta values to use, in degrees

    # Please see the Matlab documentation for hough():
    # http://www.mathworks.com/help/images/ref/hough.html
    # Your code should imitate the Matlab implementation.

    # Pay close attention to the coordinate system specified in the assignment.
    # Note: Rows of H should correspond to values of rho, columns those of theta.


    img = BW


    ### Checks

    # Binary image check
    assert len(img.shape) == 2, \
        "Input image must be a binary image"

    # RhoResolution check
    diag_len = np.ceil(np.linalg.norm(img.shape))   # Diag. length of img (max value of rho)
    assert rho_resolution < diag_len, \
        "rho_resolution must be < {0}".format(diag_len)

    # Theta check
    theta_low = int(theta_range[0])
    theta_up = int(theta_range[1])
    assert theta_up > theta_low, \
        "theta_range upper limit must be > lower limit"


    ### Initialization

    theta_arr = np.arange(theta_low, theta_up +1, step=theta_step)
    rho_arr = np.arange(-diag_len, diag_len +1, step=rho_resolution)

    edge_pts_y_arr, edge_pts_x_arr = np.nonzero(img)   # Indices of points in edge image
    cos_theta_arr = np.cos(np.deg2rad(theta_arr))   # Radians
    sin_theta_arr = np.sin(np.deg2rad(theta_arr))   # Radians
    hough_space_accum = np.zeros((len(rho_arr), len(theta_arr),), dtype=np.uint64)    # rows: rho, cols: theta


    ### Run voting procedure in Hough Space

    # We loop through thetas and accumulate the rhos (of each edge pt.) for each theta
    # This makes it easier to use the numpy bin() function to get the right indices for the rhos

    for i in range(len(theta_arr)):  # Loop through thetas
        cos_theta = cos_theta_arr[i]
        sin_theta = sin_theta_arr[i]
        rho_accum_arr = np.empty(edge_pts_x_arr.shape)   # Initialize rho array for theta value

        for j in range(len(edge_pts_x_arr)):   # Loop through edge points
            x = edge_pts_x_arr[j]
            y = edge_pts_y_arr[j]
            rho_accum_arr[j] = x*cos_theta + y*sin_theta   # rho

        bins = rho_arr
        rho_idx_arr = np.digitize(rho_accum_arr, bins)   # Get indices based on bin values from rho_arr
        for rho_idx in rho_idx_arr:
            hough_space_accum[rho_idx, i] += 1   # Vote


    return hough_space_accum, theta_arr, rho_arr














