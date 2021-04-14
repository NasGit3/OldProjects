import cv2

import numpy
import math
import sys
import os

from numpy.random import random_integers
from scipy.signal import convolve2d

import ntpath # used in leaf

# Producing Elastic Distorted images


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma
    source: https://github.com/MyHumbleSelf/cs231n/blob/master/assignment3/cs231n/data_augmentation.py

    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int
    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float

    :returns: a numpy 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = numpy.zeros((dim, dim), dtype=numpy.float16)

    # calculate the center point
    center = dim/2

    # calculate the variance
    variance = sigma ** 2

    # calculate the normalization coefficient
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2*variance

            kernel[x,y] = coeff * numpy.exp(-1. * numerator/denom)

    # normalise it
    return kernel/sum(sum(kernel))



def elastic_transform(image, kernel_dim, sigma, alpha, negated=False):
    """
    This method performs elastic transformations on an image by convolving
    with a gaussian kernel.
    NOTE: Image dimensions should be square

    :param image: the input image
    :type image: a numpy nd array
    :param kernel_dim: dimension(1-D) of the gaussian kernel
    :type kernel_dim: int
    :param sigma: standard deviation of the kernel
    :type sigma: float
    :param alpha: a multiplicative factor for image after convolution
    :type alpha: float
    :param negated: a flag indicating whether the image is negated or not
    :type negated: boolean
    :returns: a nd array transformed image
    """

    # convert the image to single channel if it is multi channel one
    if len(image.shape) == 3:
    # if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check if the image is a negated one
    if not negated:
        image = 255-image

    # check if the image is a square one
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image should be of sqaure form")

    # check if kernel dimension is odd
    if kernel_dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # create an empty image
    result = numpy.zeros(image.shape)

    # create random displacement fields
    displacement_field_x = numpy.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha
    displacement_field_y = numpy.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha

    # create the gaussian kernel
    kernel = create_2d_gaussian(kernel_dim, sigma)

    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel)
    displacement_field_y = convolve2d(displacement_field_y, kernel)

    # make the distorted image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields

    for row in xrange(image.shape[1]):
        for col in xrange(image.shape[0]):
            low_ii = row + int(math.floor(displacement_field_x[row, col]))
            high_ii = row + int(math.ceil(displacement_field_x[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
               or high_jj >= image.shape[0] - 1:
                continue

            res = image[low_ii, low_jj]/4 + image[low_ii, high_jj]/4 + \
                    image[high_ii, low_jj]/4 + image[high_ii, high_jj]/4

            result[row, col] = res

    # if the input image was not negated, make the output image also a non
    # negated one
    if not negated:
        result = 255-result

    return result



if __name__ == '__main__':


    # just call the function elastic_transform function
    # with a suitable kernel size, alpha and sigma
    # as a rule of thumb, if use sigma as a value near 6,
    # alpha 36-40, kernel size 13-15
    #
    # NOTE: the input image SHOULD be of square dimension,
    # ie no.of rows should be equal to number of cols.



    projectPath = os.getcwd()
    trainDataFolderPath = projectPath + '/mnist/train/'
    print trainDataFolderPath
    #  this line checks if the folder exist
    if not(os.path.isdir(trainDataFolderPath)):
        print trainDataFolderPath + ' is not a directory'
        sys.exit()


    # Going through each subdirectory and all the images that each subdirectory contains

    #  To get the folder list that satisfy a condition
    listOfFolders = [os.path.join(root,name)
                        for root, dirs, files in os.walk(trainDataFolderPath)
                        for name in dirs
                        if (len(name) <2)]
    print listOfFolders
    listOfFolders.sort()
    print listOfFolders

    for dir in listOfFolders:
        newFolderToCreate = trainDataFolderPath+path_leaf(dir)+'_distorted'


        if not os.path.exists(newFolderToCreate):
            os.makedirs(newFolderToCreate)
        listOfFiles = [os.path.join(root, name)
                         for root, dirs, files in os.walk(dir)
                         for name in files]
        for file in listOfFiles:

            image = cv2.imread(file)
            print path_leaf(file)
            try:
                distorted_image = elastic_transform(image, kernel_dim=13,
                                                 alpha=18,
                                                 sigma=3)
                fileToWrite = newFolderToCreate +'/d' +path_leaf(file)
                print fileToWrite

                cv2.imwrite(fileToWrite, distorted_image)

            except:
                pass



