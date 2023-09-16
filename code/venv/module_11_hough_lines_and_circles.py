import cv2 as cv
import numpy as np
import math

# func for extract coordinates of vector by HoughLines() and print it by lines()
def regular_hough(img):
    dst = cv.Canny(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 50, 200, None, 3)

    colored_dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 750 * (-b)), int(y0 + 750 * (a)))
            pt2 = (int(x0 - 750 * (-b)), int(y0 - 750 * (a)))

            cv.line(colored_dst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", colored_dst)

# func for extract coordinates of lines by HoughLinesP() and print it by lines()
def prob_hough(img):
    dst = cv.Canny(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 50, 200, None, 3)

    colored_dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(colored_dst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", colored_dst)

# func for extract coordinates of circle objects by HoughCircle() and print it by circle()
def circle_hough():
    src = cv.imread("many_circles.jpg")

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=50, param2=17,
                              minRadius=1, maxRadius=250)

    if circles is not None:
        circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(src, center, 1, (0, 0, 0), 3)
        # circle outline
        radius = i[2]
        cv.circle(src, center, radius, (0, 0, 0), 2)

    cv.imshow("Hough circles", src)


if __name__ == "__main__":
    image = cv.imread("lines.jpg")

    print("ESC - exit\nr - use HoughLines()\np - use HoughLinesP()\nc - use HoughCircles()")

    while 1:
        cv.imshow("Original image", image)

        k = cv.waitKey(0)

        if k == 27:
            break

        elif chr(k) == 'r':
            regular_hough(image)

        elif chr(k) == 'p':
            prob_hough(image)

        elif chr(k) == 'c':
            circle_hough()
