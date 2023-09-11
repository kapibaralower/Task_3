import cv2 as cv
import numpy as np


erosion_size = 0
max_elem = 2
max_kernel_size = 21
morph_op_arr = [cv.MORPH_OPEN, cv.MORPH_CLOSE, cv.MORPH_GRADIENT, cv.MORPH_TOPHAT, cv.MORPH_BLACKHAT]
explanations_arr = [["1 - 0: erosion, 1: dilatation, 2: more morph operations (trackbar 4)"],
                    ["2 - Element: 0: Rect, 1: Cross, 2: Ellipse"],
                    ["3 - Kernel size: 2n +1"],
                    ["4 - 0: Opening, 1: Closing, 2: Gradient, 3: Top Hat, 4: Black Hat"]]
title_trackbar_type_of_change = "1"
title_trackbar_element_shape = '2'
title_trackbar_kernel_size = '3'
title_trackbar_more_morph_transforms = '4'
title_of_window = "Transformed image"
title_of_trackbar_window = "Trackbars"
title_explanation_for_trackbar_window = "Explanation"

# func for select type of morphology transformation by trackbar
def type_of_change(val):
    number = cv.getTrackbarPos(title_trackbar_type_of_change, title_of_trackbar_window)

    if number == 0:
        return erosion(val)
    elif number == 1:
        return dilatation(val)
    else:
        return more_morphlogy(val)

# func for select element for morphology transformation by trackbar
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE

# func for adding erode effect on image by information from trackbar
def erosion(val):
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_of_trackbar_window)
    erosion_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_of_trackbar_window))

    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    erosion_dst = cv.erode(src, element)
    cv.imshow(title_of_window, erosion_dst)

# func for adding dilatation effect on image by information from trackbar
def dilatation(val):
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_of_trackbar_window)
    dilation_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_of_trackbar_window))

    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))

    dilatation_dst = cv.dilate(src, element)
    cv.imshow(title_of_window, dilatation_dst)

# func for adding some extend morphology transformations on image by information from trackbar
def more_morphlogy(val):
    morph_operator = cv.getTrackbarPos(title_trackbar_more_morph_transforms, title_of_trackbar_window)
    morph_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_of_trackbar_window)
    morph_elem = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_of_trackbar_window))

    element = cv.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1),
                                       (morph_size, morph_size))

    operation = morph_op_arr[morph_operator]

    morph_dst = cv.morphologyEx(src, operation, element)
    cv.imshow(title_of_window, morph_dst)

# func for adding some text from array on image
def put_text_explanation(img, title):
    for i in range(4):
        cv.putText(img, str(explanations_arr[i][0:])[2:-2],
                   [25, i*25+25], 1, 1.5, [255, 255, 255], 2, 1)

    cv.imshow(title, img)


if __name__ == "__main__":
    src = cv.imread("image.jpg")
    src = cv.resize(src, [1200, 446])
    explanation = np.zeros([300, 900], dtype=np.uint8)

    cv.namedWindow(title_of_window)
    cv.namedWindow(title_of_trackbar_window)
    cv.namedWindow(title_explanation_for_trackbar_window)
    cv.imshow("Original image", src)

    put_text_explanation(explanation, title_explanation_for_trackbar_window)

    cv.createTrackbar(title_trackbar_type_of_change, title_of_trackbar_window, 0, 2, type_of_change)
    cv.createTrackbar(title_trackbar_element_shape, title_of_trackbar_window, 0, max_elem, type_of_change)
    cv.createTrackbar(title_trackbar_kernel_size, title_of_trackbar_window, 0, max_kernel_size, type_of_change)
    cv.createTrackbar(title_trackbar_more_morph_transforms, title_of_trackbar_window, 0, 4, type_of_change)

    erosion(0)
    dilatation(0)

    cv.waitKey()
