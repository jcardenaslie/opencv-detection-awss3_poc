# https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

import pedestrian as ped

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False,
                help="path to images directory")
ap.add_argument("-v", "--video", required=False, help="path to video file")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])
while(vs.isOpened()):
    ret, image = vs.read()

    image = imutils.resize(image, width=min(600, image.shape[1]))

    image = cv2.flip(image, 1)

    image = cv2.flip(image, 0)
    
    ped.PedestrianDetectionVideo(image)

    cv2.imshow("After NMS", image)

    key = cv2.waitKey(10) & 0xFF

        # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
