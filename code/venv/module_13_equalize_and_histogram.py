import cv2 as cv
import numpy as np

hist_size = 256
hist_range = (0, 256)
hist_w = 512
hist_h = 400

# func to create histogram of simple channel image
def common_histogram(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    histogram = np.zeros((hist_h, hist_w))

    hist_coord = cv.calcHist(gray, [0], None, [hist_size], hist_range, accumulate=False)

    cv.normalize(hist_coord, hist_coord, alpha=0, beta=hist_size, norm_type=cv.NORM_MINMAX)

    for i in range(hist_size-1):
        cv.line(histogram, (i * 2, hist_h),
                (i * 2, hist_h-int(hist_coord[i][0])), (255, 0, 0), thickness=2)

    cv.imshow("Histogram of original image", histogram)
    cv.imshow("Gray image", gray)

# func to create histogram of equalized simple channel image
def equalized_histogram(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    histogram = np.zeros((hist_h, hist_w))

    gray = cv.equalizeHist(gray)

    hist_coord = cv.calcHist(gray, [0], None, [hist_size], hist_range, accumulate=False)

    cv.normalize(hist_coord, hist_coord, alpha=0, beta=hist_size, norm_type=cv.NORM_MINMAX)

    for i in range(hist_size-1):
        cv.line(histogram, (i * 2, hist_h),
                (i * 2, hist_h-int(hist_coord[i][0])), (255, 0, 0), thickness=2)

    cv.imshow("Histogram of equalized image", histogram)
    cv.imshow("Equalized image", gray)

# func to create histogram of RGB colors of 3 channel image
def colored_histogram(img):
    histogram = np.zeros((hist_h, hist_w, 3))

    bgr_planes = cv.split(img)

    b_hist = cv.calcHist(bgr_planes, [0], None, [hist_size], hist_range, accumulate=False)
    g_hist = cv.calcHist(bgr_planes, [1], None, [hist_size], hist_range, accumulate=False)
    r_hist = cv.calcHist(bgr_planes, [2], None, [hist_size], hist_range, accumulate=False)

    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_size, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_size, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_size, norm_type=cv.NORM_MINMAX)

    for i in range(1, hist_size):
        cv.line(histogram, ((i - 1) * 2, hist_h - int(b_hist[i-1][0])),
                (i * 2, hist_h - int(b_hist[i][0])), [255, 0, 0], thickness=2)

        cv.line(histogram, ((i - 1) * 2, hist_h - int(g_hist[i-1][0])),
                (i * 2, hist_h - int(g_hist[i][0])), [0, 255, 0], thickness=2)

        cv.line(histogram, ((i - 1) * 2, hist_h - int(r_hist[i-1][0])),
                (i * 2, hist_h - int(r_hist[i][0])), [0, 0, 255], thickness=2)

    cv.imshow("Histogram of colors", histogram)

if __name__ == "__main__":
    image = cv.resize(cv.imread("image.jpg"), (1200, 446))

    while 1:
        cv.imshow("Original image", image)

        k = cv.waitKey(0)

        if k == 27:
            break

        elif chr(k) == 'h':
            common_histogram(image)

        elif chr(k) == 'e':
            equalized_histogram(image)

        elif chr(k) == 'c':
            colored_histogram(image)