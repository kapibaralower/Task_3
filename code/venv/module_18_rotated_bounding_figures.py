import cv2 as cv
import numpy as np
from random import randint


def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    min_rect = [None] * len(contours)
    min_ellipse = [None] * len(contours)

    for i, c in enumerate(contours):
        min_rect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            min_ellipse[i] = cv.fitEllipse(c)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i, c in enumerate(contours):
        color = (randint(0, 256), randint(0, 256), randint(0, 256))

        cv.drawContours(drawing, contours, i, color)

        if c.shape[0] > 25:
            cv.ellipse(drawing, min_ellipse[i], color, 2)

    box = cv.boxPoints(min_rect[i])
    box = np.intp(box)

    cv.drawContours(drawing, [box], 0, color)

    cv.imshow('Contours', drawing)

if __name__ == "__main__":
    src = cv.imread("random_figures.jpg")

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    src_gray = cv.blur(src_gray, (3, 3))

    source_window = 'Source'

    cv.namedWindow(source_window)

    cv.imshow(source_window, src)
    max_thresh = 255

    thresh = 100

    cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)

    thresh_callback(thresh)
    cv.waitKey()