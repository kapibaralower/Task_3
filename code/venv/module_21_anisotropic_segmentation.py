import cv2 as cv
import numpy as np
import argparse

W = 30  # window size is WxW
C_Thr = 0.43  # threshold for coherency
LowThr = 35  # threshold1 for orientation, it ranges from 0 to 180
HighThr = 57  # threshold2 for orientation, it ranges from 0 to 180


def calcGST(inputIMG, w):
    img = inputIMG.astype(np.float32)
    # GST components calculation (start)
    imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
    imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, 3)

    imgDiffXY = cv.multiply(imgDiffX, imgDiffY)
    imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv.multiply(imgDiffY, imgDiffY)

    J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w, w))
    J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w, w))
    J12 = cv.boxFilter(imgDiffXY, cv.CV_32F, (w, w))
    # GST components calculations (stop)
    # eigenvalue calculation (start)
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv.multiply(tmp2, tmp2)
    tmp3 = cv.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)

    lambda1 = 0.5 * (tmp1 + tmp4)  # biggest eigenvalue
    lambda2 = 0.5 * (tmp1 - tmp4)  # smallest eigenvalue
    # eigenvalue calculation (stop)
    # Coherency calculation (start)
    imgCoherencyOut = cv.divide(lambda1 - lambda2, lambda1 + lambda2)
    # Coherency calculation (stop)
    # orientation angle calculation (start)
    imgOrientationOut = cv.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
    imgOrientationOut *= 0.5
    # orientation angle calculation (stop)

    return imgCoherencyOut, imgOrientationOut


if __name__ == "__main__":
    imgIn = cv.imread("anisotropic.jpg", cv.IMREAD_GRAYSCALE)
    cv.imshow("Original", imgIn)

    imgCoherency, imgOrientation = calcGST(imgIn, W)

    _, imgCoherencyBin = cv.threshold(imgCoherency, C_Thr, 255, cv.THRESH_BINARY)
    _, imgOrientationBin = cv.threshold(imgOrientation, LowThr, HighThr, cv.THRESH_BINARY)

    imgBin = cv.bitwise_and(imgCoherencyBin, imgOrientationBin)

    imgCoherency = cv.normalize(imgCoherency, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    imgOrientation = cv.normalize(imgOrientation, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    cv.imshow('result', np.uint8(0.5 * (imgIn + imgBin)))

    cv.imshow('Coherency', imgCoherency)

    cv.imshow('Orientation', imgOrientation)

    cv.waitKey(0)