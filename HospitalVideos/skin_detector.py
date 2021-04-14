# import the necessary packages
from pyimagesearch import imutils
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
cap = cv2.VideoCapture('~/myVideo.mp4')
print cap.isOpened()

""" 
    define the upper and lower boundaries of the HSV pixel
    intensities to be considered 'skin'. Upper and lower limits can be tuned
"""
lower = np.array([164, 72, 93], dtype="uint8")
upper = np.array([180, 145, 170], dtype="uint8")

while True:
    ret, frame = cap.read()
    if ret is False:
		break
""" 
    resize the frame, convert it to the HSV color space,
    and determine the HSV pixel intensities that fall into
    the specified upper and lower boundaries
"""
    imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    skinMask = cv2.erode(skinMask, kernel_erosion, iterations = 1)
    skinMask = cv2.dilate(skinMask, kernel_dilation, iterations = 2)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    cv2.imshow('frame',frame)

    cv2.imshow('skin',skin)
    if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()