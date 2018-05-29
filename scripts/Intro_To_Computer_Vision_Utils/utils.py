import cv2 as cv


def print_image_details_and_show(img, chart_title, show=True):
    print("Image: " + chart_title)
    print(img.dtype)
    print(img.shape)

    if len(img.shape) == 2:
        print("grayscale")
    elif len(img.shape) == 3:
        print("colour")

    print()

    if show:
        cv.imshow(chart_title, img)
        cv.waitKey(0)