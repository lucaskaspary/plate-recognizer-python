import cv2
from matplotlib import pyplot
import numpy
import imutils
import torch
import easyocr


if __name__ == '__main__':

    img = cv2.imread('./img/plate2.png')
    pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pyplot.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pyplot.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    pyplot.show()

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    pyplot.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    pyplot.show()

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10 , True)
        if len(approx) == 4:
            location = approx
            break

    mask = numpy.zeros(gray.shape, numpy.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    pyplot.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    pyplot.show()

    (x,y) = numpy.where(mask==255)
    (x1, y1) = (numpy.min(x), numpy.min(y))
    (x2, y2) = (numpy.max(x), numpy.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    pyplot.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    pyplot.show()

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    print(result)
