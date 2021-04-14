__author__ = 'Nasrin'

"""
  The background subtraction is performed by comparing (subtracting) each frame 
  with the background frame (here the last frame) 
  then performing different threshold techniques on the result of comparision.
  finally apply morphology (here open) to get rid of the ripples
"""

import numpy as np
import cv2

print cv2.__version__
cap = cv2.VideoCapture('~/myVideo.mp4')
# Print cap.isOpened()   # True = read video successfully. False - fail to read video.

# Getting the number of frames
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print "The number of frames: ", frame_count

# Getting the last frame as your background
last_frame = frame_count-1
cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)


# if not cap.isOpened():              # check if the video exists
#     print "Could not open the file"
#     pass
# Reading the last frame, then converting the RGB to gray-scale
ret, BackGround_frame = cap.read()
gray_BGframe = cv2.cvtColor(BackGround_frame, cv2.COLOR_BGR2GRAY)


''' Set the video object back to the first frame. 2: cv2.CAP_PROP_POS_AVI_RATIO. 
   This is the relative position of the video
   file. Also,  0: first frame  This is done in cap.set(2,0)
'''
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

kernel = np.ones((3, 3), np.uint8)  # Kernel size can be tuned
while cap.isOpened():

    ret, frame = cap.read()

    # break the loop, once we get to the very last frame
    if ret is False:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayDiff = cv2.absdiff(gray_frame, gray_BGframe)

    thresh = cv2.threshold(grayDiff, 70, 255, cv2.THRESH_BINARY)[1]  # lower and upper threshold values can be tuned
    kernel_dilate = np.ones((13, 13), np.uint8)   # Kernel size can be tuned
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_dilate)

    AllFrames = np.hstack((thresh, gray_frame))
    cv2.imshow('Orig+MedianBg+Diff', AllFrames)
    k = cv2.waitKey(25) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

