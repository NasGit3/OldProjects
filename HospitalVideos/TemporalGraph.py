__author__ = 'Nasrin'

""" Creating a temporal graph on top of 
    the frames of a video
    Please make sure to cite the following if you use this code:
    "Graph-based classification of healthcare provider activity"
    Link of the paper: https://ieeexplore.ieee.org/abstract/document/7869577

"""
import os
import sys
import ntpath
import pandas as pd

numArg = len(sys.argv)
import numpy as np
import cv2
from matplotlib import pyplot as plt
import timeit

import imutils
import math
from numpy import linalg as LA
from numpy.linalg import eigh
from matplotlib import pyplot as plt

start = timeit.default_timer()
ListOfHistograms = []


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)   # head gives the path and tail gives the file name (here the video name)


# Name of the Videos that were used for this work contained room number,
# weekday, month, day, TimeHour, Duration, and year
def parseFileName(fileName):
    RoomNum = fileName[0:4]
    WeekDay = fileName[7:10]
    Month = fileName[11:14]
    Day = fileName[15:17]
    TimeHour = fileName[18:20] + 'h'
    TimeMin = fileName[21:23] + 'm'
    Year = fileName[-8:-4]
    return RoomNum, Year, Month, WeekDay, Day, TimeHour, TimeMin


# FrametoArray returns all the frames inside an array which is transformedFrames
def FrametoArray(cap):
    print 'inside process Video Capture'

    dim = (440, 220)
    rowNum = dim[1]
    colNum = dim[0]

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print frame_count
# creating an array to save all the transformed frames.
    # Each column of the TransformedFrames contains all the pixels of one frame
    if frame_count % 2 == 0:
        transformedFrames = np.zeros((rowNum*colNum,int(frame_count/2)))

    else:
        transformedFrames = np.zeros((rowNum*colNum,int(frame_count/2+1)))


    frameCount = 0
    frameNum = 0
    while True:
        ret, frame = cap.read()

        if ret is False:
            break

        if frameCount % 2 != 0:
            frameCount += 1
            continue

        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        transformedFrames[:, frameNum] = np.reshape(hsv_image[:, :, 2], (rowNum*colNum,))
        frameNum += 1
        frameCount += 1
        print frameNum
    print transformedFrames.shape
    return transformedFrames

def FramestoGraph(frames):

    patchsize = 11
    patchPixels = patchsize*patchsize
    transPixels = np.zeros((frames.shape[0]*frames.shape[1]))
    sigmaIntensitySqu = (patchPixels*255**2)/2
    cntTransPixels = 0

    for i in range(0, frames.shape[0], patchPixels):
        intensityWeight = np.zeros((frames.shape[1]-1))
        cntWeight = 0
        for j in range(0, frames.shape[1]-1):
            intensityWeight[cntWeight] = np.exp(-(LA.norm(frames[i:i+patchPixels, j] -
                                                frames[i:i+patchPixels, j+1]))**2/(2*sigmaIntensitySqu))
            cntWeight += 1

        A = np.zeros((frames.shape[1], frames.shape[1]))

        for m in range(0, intensityWeight.shape[0]):
            A[m, m+1] = intensityWeight[m]
            A[m+1, m] = A[m, m+1]

        D = np.zeros((frames.shape[1],frames.shape[1]))

        # Filling the Degree matrix
        for n in range(0, frames.shape[1]):
            weightsSum = 0
            for p in range(0, frames.shape[1]):
                weightsSum += A[n, p]

            D[n, n] = weightsSum

        L = D-A
        vals, vecs = eigh(L)
        minRow = i
        maxRow = i+patchPixels
        for k in range(minRow,maxRow):
            transPixels[cntTransPixels:cntTransPixels+frames.shape[1]] = \
                np.reshape(np.dot(vecs, frames[minRow, :]), (frames.shape[1],))
            cntTransPixels += frames.shape[1]

    values, bin_edges = np.histogram(transPixels, bins=60, range=[-600, 600])   #bins=30, range=[-600,600]

    width = (bin_edges[1] - bin_edges[0])
    bin_min= bin_edges[np.where(values == np.max(values))]
    bin_max= bin_min + width
    new_val = transPixels[(transPixels < bin_min) | (transPixels > bin_max)]

    values, bin_edges = np.histogram(new_val, bins=59, range=[-600, 600])
    normalizedHistogram = values
    return normalizedHistogram


# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

if numArg < 2:
    videosFolder = os.getcwd()  # getting the working directory
    print 'No path for video has been provided --> default video folder  = videosFolder :  ' + videosFolder
elif numArg == 2:
    videosFolder = sys.argv[1]
    if not(os.path.isdir(videosFolder)):
        print videosFolder + ' is not a folder'
        sys.exit()
    else:
        print 'Selected video folder is : ' + videosFolder
else:
    print 'usage  :' + sys.argv[0] + ' /path/to/video/files '
    sys.exit()


# This gets the file names that end with .mp4 from videosFolder. root is the working directory
videoFilePathList = [os.path.join(root, name)
                    for root, dirs, files in os.walk(videosFolder)
                    for name in files
                    if (name.endswith('.mp4'))]
videoFilePathList.sort()

for vFile in videoFilePathList[0:]:
    print "Analyzing file: "  + path_leaf(vFile)
    RoomNum, Year, Month, WeekDay, Day, TimeHour, TimeMin=parseFileName(path_leaf(vFile))
    if not(RoomNum.isdigit()):
        print "Unexpected room number -->  should be numeric"
        continue

    cap = cv2.VideoCapture(vFile)
    print "Opening file successful?  " + str(cap.isOpened())
    videoDic = {}
    videoDic['VideoName']= path_leaf(vFile)

    (allFrames) = FrametoArray(cap)
    (histogram) = FramestoGraph(allFrames)

    videoDic['VideoHistogram'] = histogram
    ListOfHistograms.append(videoDic)


df = pd.DataFrame.from_records(ListOfHistograms, index='VideoName')['VideoHistogram'].apply(pd.Series)
df.to_csv('TempGraph_Histbins.csv')


stop = timeit.default_timer()
print stop - start

