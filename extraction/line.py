import cv2
import pandas as pd
import numpy as np
import math 
import imutils

def line_angle(line):
    """Return the angle between line and horizontal line (using arc tangent)

    Paramaters:
    -----------
    line : list of integers
        contain [x1,y1,x2,y2]
    
    Return:
    -----------
    (float)
        angle between line and horizontal line
    """
    return math.atan2(line[1] - line[3], line[0] - line[2])*180/math.pi

def switchHigherLowerVertical(data):
    """compare 2 value then return in specific order

    Paramaters:
    -----------
    data : dictionary 
        contain key of 'y1' and 'y2' with value of integer
    """
    if data['y1'] >= data['y2']:
        return data['y1'], data['y2']
    else:
        return data['y2'], data['y1']

def switchHigherLowerHorizontal(data):
    """compare 2 value then return in specific order

    Paramaters:
    -----------
    data : dictionary 
        contain key of 'x1' and 'x2' with value of integer
    """
    if data['x1'] >= data['x2']:
        return data['x2'], data['x2']
    else:
        return data['x1'], data['x2']

def sort_contours(cnts, method="left-to-right"):
    """Sort contours of a input image, with the option of top-to-bottom, bottom-to-top, left-to-right, right-to-left

    Parameters
    --------
    cnts : list of integer
        contours of input image, refer to return contours of cv2.findcontours
    method : str 
        method of sorting the contours, top-to-bottom, bottom-to-top, left-to-right, right-to-left

    Returns:
    ----------
    cnts : list of integer
        contours of result image
    boundingBoxes : list of list of integer 
        rectangle bounding boxes of result 
    """ 
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def check_B_inside_A(A, B):
    """check 2 bounding boxes wether the first one inside the second one or not

    Paramaters:
    -----------
    A : list of integer 
        first bounding box
    B: list of integer
        second bounding box
    
    Returns:
    ------------
    Boolean True or Boolean False
    """
    # If rectangle B is inside rectangle A
    # bottomA <= topB
    if((A[0] <= B[0]) and (A[0]+A[2] >= B[0]+B[2]) and (A[1] <= B[1]) and (A[1]+A[3] >= B[1]+B[3])):
        return True
    else:
        return False