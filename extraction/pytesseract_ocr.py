import pytesseract
from pytesseract import Output
import cv2
import pandas as pd

def get_intersection(bb1, bb2):
    """Algorithm to get the intersection between 2 bounding boxes, will return negative value if both bounding boxes completely separated.

    Parameters
    ----------
    bb1 : list of integer 
        boundingboxes number 1
    bb2 : list of integer 
        boundingboxes number 1

    Returns
    ----------
    intersection_result : float
        area of intersection, will return negative value if both bounding boxes completely separated
    """
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])
    if ((x_right - x_left) < 0 and (y_bottom - y_top) < 0):
        intersection_area = (x_right - x_left) * (y_bottom - y_top) * (-1)
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    if(bb2_area == 0):
        intersection_result = -1
    else:
        intersection_result = intersection_area / float(bb2_area)
    return intersection_result

def inside_box_all(boundb, innerb):
    """Algorithm to return wether all the bounding box of boundb is inside the bounding box of innerb .

    Parameters
    ----------
    boundb : list of integer list of integer 
        multi boundingboxes 
    innerb : list of integer 
        boundingboxes 

    Returns
    ----------
    result_final : boolean
        True if all boundingboxes in boundb is inside the bounding box in innerb
    """
    result = []
    for i in range(len(boundb)):
        if (boundb[i][1] <= innerb[1] and boundb[i][0] <= innerb[0] and innerb[0] <= boundb[i][0]+boundb[i][2] and innerb[1] <= boundb[i][1]+boundb[i][3]):
            result.append(True)
        else:
            result.append(False)
    result_final = result[0]
    for i in range(len(result)):
        result_final = result_final or result[i]
    return(result_final)

def combine_lines_and_bb(x):
    """Combine row of column text, 
    minimum value of column left,
    minimum value of column top
    maximum value of column height,
    sum of column height 
    and return in as pandas series
    
    """
    return pd.Series(dict(
        left=x['left'].min(), top=x['top'].min(), width=x['width'].max(), 
        height=x['height'].sum(), lines_combine="%s" % ' '.join(x['text'])
        ))

def combine_lines_comma_separated_and_bb(x):
    """Combine row of column lines_combine with comma separated, 
    minimum value of column left,
    minimum value of column top
    maximum value of column height,
    sum of column height 
    and return in as pandas series
    
    """
    return pd.Series(
        dict(
        left=x['left'].min(), top=x['top'].min(), 
        width=x['width'].max(), height=x['height'].sum(), 
        lines_combine="%s" % ', '.join(x['lines_combine'])))