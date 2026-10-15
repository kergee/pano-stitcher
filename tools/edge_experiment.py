"""Rejected nearest-color extension experiment; not a production mode."""
import cv2
import numpy as np


def extend_valid_colors(image, mask):
    """Nearest valid color extension before pyramid filtering; support mask stays unchanged.

    Inspired by support-boundary handling; not AutoStitch's 0.5 two-axis formula.
    C++: cv::distanceTransform(..., DIST_LABEL_PIXEL).
    """
    valid=mask>0
    if not valid.any():return image
    if valid.all():return image
    _,labels=cv2.distanceTransformWithLabels((~valid).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    colors=image[valid]
    extended=colors[np.clip(labels-1,0,len(colors)-1)]
    extended[valid]=image[valid]
    return extended
