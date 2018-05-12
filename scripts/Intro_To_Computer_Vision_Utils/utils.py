import cv2 as cv


def print_image_details_and_show(img, chart_title='image', show=True):
    print(img.dtype)
    print(img.shape)

    if show:
        cv.imshow(chart_title, img)
        cv.waitKey(0)