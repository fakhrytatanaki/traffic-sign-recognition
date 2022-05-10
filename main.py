import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def getBoundaries(contours,hierarchy,minHierarchyLevel): 
    boundaries = []

    for i,c in enumerate(contours):
        if hierarchy[0][i][3] >= minHierarchyLevel:
            x = [u[0][0] for u in c]
            y = [u[0][1] for u in c]
            boundaries+=[[(np.min(x),np.max(y)),(np.max(x),np.min(y))]]

    return boundaries


def drawBoundaries(boundaries,img,colour):
    img_boundaries = np.copy(img)
    for b in boundaries:
        cv.rectangle(img_boundaries, b[0], b[1], colour)
    return img_boundaries

cap = cv.VideoCapture(0) #opens the webcam if exists

for i in range(3600): #show up to 3600 frames
    _,img = cap.read() #read from webcam
    imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imggr = cv.GaussianBlur(imggr, (7,7), 1.7)
    #imggr = cv.medianBlur(imggr, 7)
    edges = cv.Canny(imggr, 30, 70)
    contours,hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    b = getBoundaries(contours, hierarchy, -1)
    img_boundaries = drawBoundaries(b,img, (255,0,0))

    cv.imshow('test', img_boundaries)
    k = cv.waitKey(20)  
    if (k==ord('q')): #quit the loop if the key [q] is pressed
        break





