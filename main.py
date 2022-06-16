import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import shapes as s
from glob import glob
from mltools import SVMOneVsRest,NeuralNetworkModel


def normalize(arr):
    vmin = np.min(arr)
    vmax = np.max(arr)

    return (arr-vmin)/(vmin-vmax)

def histogramAvg(datasetDirectory):
        imagesByLabel = []
        labelSet = [f for f in os.listdir(datasetDirectory) if not f.startswith('.')]

        for l in labelSet:
            imagesByLabel.append( [cv.imread(g) for g in glob(os.path.join(datasetDirectory,l,'*.png')) ] )

        histAvgs = []

        for images in imagesByLabel:
            histAvgs.append(normalize(np.average([np.histogram(img.ravel(),256,[0,256])[0] for img in images],axis=0)))

        # histAvgsDict = {}
        # for i,l in enumerate(labelSet):
        #     histAvgsDict[l] = histAvgs[i]

        return histAvgs


G_SHAPES = s.getShapes()

G_CANNY_MIN = 100
G_CANNY_MAX = 250

G_GAUSSIAN_KERN_SIZE = (3,3)
G_GAUSSIAN_STD_DEV = 1.5

G_SHAPES_DETECT_ERR_MAX = 0.1

G_POLY_APPROX_ERR = 0.0005

# SVMOneVsRestModel = SVMOneVsRest()
# SVMOneVsRestModel.loadDataset('small_dataset') #it might take a while to load the big dataset, for testing purposes, we choose 'small_dataset for not'

nnm  = NeuralNetworkModel('./dataset')


# AvgHists = histogramAvg('small_dataset')





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
            

        
def similarHistRoutine(imgIn,boundingRects,avgHists,histErr):

        filteredBoundingRects = []
        minerrs = []

        for x,y,w,h in boundingRects:
            cr = imgIn[y:y+h,x:x+w]
            hist = np.histogram(cr.ravel(),256,[0,256])[0]
            errForEachHist = np.array([ np.sum(np.abs(normalize(hist) - avgHist)) for avgHist in avgHists ])
            minerrs.append(np.min(errForEachHist))
            if np.min(errForEachHist) < histErr:
                filteredBoundingRects.append((x,y,w,h))

        if minerrs:
            print(np.min(minerrs))

        return filteredBoundingRects








def removeSmallContours(contours,smin=(100,100)):
    contoursFiltered = []
    wmin,hmin = smin
    for c in contours:
        _,_,w,h = cv.boundingRect(c)
        if w >= wmin or h >= hmin:
            contoursFiltered.append(c)
    return contoursFiltered




def shapeDetectionRoutine(contours,err):


    # detectedShapes = {}

    # for shapeName in G_SHAPES:
    #     detectedShapes[shapeName] = 0 #creating a table for counting number of each shape detected, initially zeros


    #contours = np.array([cv.approxPolyDP(c,G_POLY_APPROX_ERR , True) for c in contours])
    contours = removeSmallContours(contours)
    detected = detectMatchingShapes(contours, G_SHAPES,err)
    boundingRects = []

    for i,shapeName in enumerate(detected):
        if shapeName!='unknown':
            # detectedShapes[shapeName]+=1
            boundingRects.append(cv.boundingRect(contours[i]))



    # for shapeName in detectedShapes:
    #     print(f"{shapeName}(s) detected : {detectedShapes[shapeName]}")



    return boundingRects

def machineLearningPredicitonRoutine(imgIn,boundingRects):

    if len(boundingRects)==0:
        return []
    imageCrops = []
    for x,y,w,h in boundingRects:
        imageCrops.append(imgIn[y:y+h,x:x+w])

    #predictedLabels = SVMOneVsRestModel.predict(imageCrops)
    predictedLabels = nnm.predict(imageCrops)

    return predictedLabels


def drawLabels(imgIn,boundingRects,labels):
    assert(len(boundingRects)==len(labels))
    img_labels = np.copy(imgIn)

    for i,l in enumerate(labels):
        if l:
            x,y,w,h = boundingRects[i]
            cv.putText(img_labels, l, (x,y), cv.FONT_HERSHEY_COMPLEX,1, (255,255,0), 2, cv.LINE_AA)
            cv.rectangle(img_labels, (x,y), (x+w,y+h), (255,0,0))
    
    return img_labels



cap = cv.VideoCapture(0) #opens the webcam if exists


for i in range(3600): #show up to 3600 frames
     _,img = cap.read() #read from webcam
     imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     imggr = cv.GaussianBlur(imggr, G_GAUSSIAN_KERN_SIZE, G_GAUSSIAN_STD_DEV)
     edges = cv.Canny(imggr, G_CANNY_MIN, G_CANNY_MAX)
     contours,hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
     boundingRects = shapeDetectionRoutine(contours, G_SHAPES_DETECT_ERR_MAX)
     #boundingRects = similarHistRoutine(img, boundingRects, AvgHists, 60)
     labels = machineLearningPredicitonRoutine(img, boundingRects)
     resimg = drawLabels(img, boundingRects, labels)

     #img_shapes_detected,edges = shapeDetectionRoutine(img,G_SHAPES_DETECT_ERR_MAX)
     cv.imshow('test', resimg)
     k = cv.waitKey(20)  
     if (k==ord('q')): #quit the loop if the key [q] is pressed
         break





# imgTestData = [cv.imread(g) for g in glob('./dataset/stop/*.png')]


# for i in imgTestData:
#     img = i
#     img_shapes_detected,edges = shapeDetectionRoutine(img,G_SHAPES_DETECT_ERR_MAX)
#     cv.imshow('test', img_shapes_detected)
#     k = cv.waitKey(1000)  
#     if (k==ord('q')): #quit the loop if the key [q] is pressed
#         break
