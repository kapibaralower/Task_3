import cv2 as cv
import numpy as np

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = "Type"
trackbar_value = "Value"
window_name = "Threshold"

# func for applying threshold filters and show result
def threshold(val):
    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)

    _, dst = cv.threshold(image, threshold_value, max_binary_value, threshold_type)

    cv.imshow(window_name, dst)


if __name__ == "__main__":
    image = cv.imread("image.jpg")

    cv.namedWindow(window_name)
    #Explanation to threshold type
    print("Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted")

    # Create Trackbar to choose Threshold type
    cv.createTrackbar(trackbar_type, window_name, 0, max_type, threshold)
    # Create Trackbar to choose Threshold value
    cv.createTrackbar(trackbar_value, window_name, 0, max_value, threshold)

    threshold(0)

    cv.waitKey(0)
    cv.destroyAllWindows()