import cv2
import numpy as np
import datetime
from imutils.object_detection import non_max_suppression

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def PedestrianDetectionVideo(image, color=(0, 255, 0)):
    global cv2
    
    orig = image.copy()

    (rects, weights) = (rects, weights) = hog.detectMultiScale(
        image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    text = ImageProcessing(cv2, rects, orig, image, color)

    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	
    text = "False"
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), color, 2)
        text = "True"
   
    response = 1 if text == "True" else 0
    
    return response