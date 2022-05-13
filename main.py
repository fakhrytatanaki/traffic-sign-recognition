import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import shapes as s
from glob import glob

G_SHAPES = s.getShapes()

G_CANNY_MIN = 100
G_CANNY_MAX = 250

G_GAUSSIAN_KERN_SIZE = (3,3)
G_GAUSSIAN_STD_DEV = 1.5

G_SHAPES_DETECT_ERR_MAX = 0.1

G_POLY_APPROX_ERR = 0.0005



def detectMatchingShapes(contours,shapes,epsilon):
    labels = list(shapes.keys())
    shapePoints = list(shapes.values())

    detected = []

    for i,c in enumerate(contours):
        errForEveryShape = []

        for sp in shapePoints:
            errForEveryShape.append(cv.matchShapes(sp, c, cv.CONTOURS_MATCH_I3, 1)) #calculate the similarity of the contour with every shape

        minErrShape = np.nanargmin(errForEveryShape) #take the one that is the most similar (least error)
        if errForEveryShape[minErrShape] > epsilon: #if error exceeds the threshold, consider it an unknown shape
            detected.append('unknown')
        else:
            detected.append(labels[minErrShape])

    return detected
            




def removeSmallContours(contours,smin=(20,20)):
    contoursFiltered = []
    wmin,hmin = smin
    for c in contours:
        _,_,w,h = cv.boundingRect(c)
        if w >= wmin or h >= hmin:
            contoursFiltered.append(c)
    return contoursFiltered



def shapeDetectionRoutine(imgIn,err):

    imggr = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
    imggr = cv.GaussianBlur(imggr, G_GAUSSIAN_KERN_SIZE, G_GAUSSIAN_STD_DEV)
    edges = cv.Canny(imggr, G_CANNY_MIN, G_CANNY_MAX)

    detectedShapes = {}

    for shapeName in G_SHAPES:
        detectedShapes[shapeName] = 0 #creating a table for counting number of each shape detected, initially zeros

    contours,hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #contours = np.array([cv.approxPolyDP(c,G_POLY_APPROX_ERR , True) for c in contours])
    contours = removeSmallContours(contours)
    detected = detectMatchingShapes(contours, G_SHAPES,err)

    for i,shapeName in enumerate(detected):
        if shapeName!='unknown':
            detectedShapes[shapeName]+=1
            x,y,w,h = cv.boundingRect(contours[i])
            cv.putText(imgIn, shapeName, (x,y), cv.FONT_HERSHEY_COMPLEX,1, (255,255,0), 2, cv.LINE_AA)
            cv.rectangle(imgIn, (x,y), (x+w,y+h), (255,0,0))


    for shapeName in detectedShapes:
        print(f"{shapeName}(s) detected : {detectedShapes[shapeName]}")



    return imgIn,edges

# cap = cv.VideoCapture(0) #opens the webcam if exists


# for i in range(3600): #show up to 3600 frames
#      _,img = cap.read() #read from webcam
#      img_shapes_detected,edges = shapeDetectionRoutine(img,G_SHAPES_DETECT_ERR_MAX)
#      cv.imshow('test', img_shapes_detected)
#      k = cv.waitKey(20)  
#      if (k==ord('q')): #quit the loop if the key [q] is pressed
#          break





imgTestData = [cv.imread(g) for g in glob('./dataset/stop/*.png')]


for i in imgTestData:
    img = i
    img_shapes_detected,edges = shapeDetectionRoutine(img,G_SHAPES_DETECT_ERR_MAX)
    cv.imshow('test', img_shapes_detected)
    k = cv.waitKey(1000)  
    if (k==ord('q')): #quit the loop if the key [q] is pressed
        break
