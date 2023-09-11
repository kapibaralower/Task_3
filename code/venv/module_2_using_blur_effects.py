import cv2 as cv

if __name__ == "__main__":

    image = cv.imread("image.jpg")

    # resize the original image for easier placement of other images
    image = cv.resize(image, [800, 332])

    blured = cv.blur(image, (1, 100))

    gaussian_blur = cv.GaussianBlur(image, (1, 101), 0)

    median_blur = cv.medianBlur(image, 13)

    cv.imshow("original", image)
    cv.imshow("blured", blured)
    cv.imshow("gaussian_blur", gaussian_blur)
    cv.imshow("median_blur", median_blur)

    cv.waitKey(0)
    cv.destroyAllWindows()

    image = cv.imread("image.jpg")
    image = cv.resize(image, [1200, 498])

    bilateral_filter = cv.bilateralFilter(image, 25, 100, 30)

    cv.imshow("original", image)

    # put the text on image with "bilateral filter"
    cv.putText(bilateral_filter, "by: kapibaralower", [400, 420], 2, 1.7,
               [255, 255, 255], 2, 1)

    cv.imshow("bilateral_filter", bilateral_filter)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # using all filters on one image
    blured = cv.GaussianBlur(cv.bilateralFilter(cv.medianBlur(blured, 13),
                                                25, 100, 30),
                             (101, 99), 0)

    cv.imshow("all in one", blured)

    cv.waitKey(0)
