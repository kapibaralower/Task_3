import cv2 as cv

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Thresh'
ratio = 3
kernel_size = 3

def CannyThreshold(val):
    low_threshold = val

    img_blur = cv.blur(image_gray, (3, 3))

    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)

    mask = detected_edges != 0

    dst = image * (mask[:, :, None].astype(image.dtype))

    cv.imshow(window_name, detected_edges)


if __name__ == "__main__":
    image = cv.imread("image.jpg")

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cv.namedWindow(window_name)
    cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)

    CannyThreshold(0)
    cv.waitKey(0)
