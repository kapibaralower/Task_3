import cv2 as cv

if __name__ == "__main__":
    image = cv.resize(cv.imread("image.jpg"), [1200, 446])

    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    image = cv.GaussianBlur(image, (3, 3), 0)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta)

    print("ESC - exit\nf - Sharr gradient of y\ns - Sobel gradient of y")

    while 1:
        cv.imshow("Original image", image)

        k = cv.waitKey(0)

        if k == 27:
            break
        elif chr(k) == 'f':
            grad_y = cv.Scharr(gray,ddepth,0,1)
        elif chr(k) == 's':
            grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        cv.imshow("Outlines", grad)
