import cv2 as cv
import numpy as np


def Hist_and_Backproj(val):
    bins = val

    histSize = max(bins, 2)

    ranges = [0, 180]

    hist = cv.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    backproj = cv.calcBackProject([hue], [0], hist, ranges, scale=1)

    cv.imshow('BackProj', backproj)

    w = 400
    h = 400
    bin_w = int(round(w / histSize))
    histImg = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(bins):
        cv.rectangle(histImg, (i * bin_w, h), ((i + 1) * bin_w, h - int(np.round(hist[i][0] * h / 255.0))), (0, 0, 255),
                     cv.FILLED)

    cv.imshow('Histogram', histImg)


if __name__ == "__main__":
    src = cv.resize(cv.imread("image.jpg"), (1200, 446))

    trackbar_window = np.empty((34, 800))

    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

    ch = (0, 0)

    hue = np.empty(hsv.shape, hsv.dtype)

    cv.mixChannels([hsv], [hue], ch)

    window_image = 'Source image'
    trackbar_title = "Trackbar"

    cv.imshow(trackbar_title, trackbar_window)

    bins = 25

    cv.createTrackbar('* Hue bins: ', trackbar_title, bins, 180, Hist_and_Backproj)

    Hist_and_Backproj(bins)

    cv.imshow(window_image, src)
    cv.waitKey()