
"""
    Detect human skin tone and draw a boundary around it.
    Useful for gesture recognition and motion tracking.
    Inspired by: http://stackoverflow.com/a/14756351/1463143
    Date: 08 June 2013
"""

import cv2
import numpy

# Constants for finding range of skin color in YCrCb. The upper and lower limits can be tuned
min_YCrCb = numpy.array([0, 133, 77], numpy.uint8)
max_YCrCb = numpy.array([255, 173, 127], numpy.uint8)

videoFrame = cv2.VideoCapture('~/myVideo.mp4')
print videoFrame.isOpened()   # True = read video successfully. False - fail to read video.

width = videoFrame.get(cv2.CAP_PROP_FRAME_WIDTH)
print "The width of frames: ", width
height = videoFrame.get(cv2.CAP_PROP_FRAME_HEIGHT)
print "The height of frames: ", height

frame_count = videoFrame.get(cv2.CAP_PROP_FRAME_COUNT)
print "The number of frames: ", frame_count

# Getting the last frame as your background
videoFrame.set(2, frame_count-1)
last_frame = frame_count-1

ret, BackGround_frame = videoFrame.read()
YCR_BGframe = cv2.cvtColor(BackGround_frame, cv2.COLOR_BGR2YCR_CB)
cropped_BGframe = YCR_BGframe[.25*height:.75*height, .25*width:.75*width]
videoFrame.set(2,0)

# Process the video frames
keyPressed = -1
while True:
    # Grab video frame, decode it and return next video frame
    ret, sourceImage = videoFrame.read()
    if ret is False:
        break
    if videoFrame.get(cv2.CAP_PROP_POS_FRAMES) == last_frame:
        break

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)

    cropped = imageYCrCb[.25*height:.75*height, .25*width:.75*width]
    img_diff = cv2.subtract(cropped, cropped_BGframe)
    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(img_diff, min_YCrCb, max_YCrCb)

    # Do contour detection on skin region
    _, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contour on the source image
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(sourceImage, contours, i, (0, 255, 0), 3)  

    # Display the source image
    cv2.imshow('Original', sourceImage)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyWindow('Original')
videoFrame.release()



