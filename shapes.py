import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import json

vecNorm = lambda x,y=0:np.sqrt(np.sum((x-y)**2))

def generateRegularPolygonInt(nsides,angleOffset=0,r=300):
    #generates integer coordinates of a regular polygon with a given side length (r)
    #nsides : number of sides
    #angleOffset : rotation of the shape

    coords = np.zeros((nsides+1,2),dtype=np.int32)

    for i,theta in enumerate(np.linspace(0,2*np.pi,num=nsides+1)):
            coords[i] = [int(r*np.cos(angleOffset + theta)),int(r*np.sin(angleOffset + theta))]

    return coords;

def generateRegularPolygon(nsides,angleOffset=0):
    #generates coordinates of a regular polygon
    #nsides : number of sides
    #angleOffset : rotation of the shape

    coords = np.zeros((nsides+1,2))

    for i,theta in enumerate(np.linspace(0,2*np.pi,num=nsides+1)):
            coords[i] = [np.cos(angleOffset + theta),np.sin(angleOffset + theta)]

    return coords;

def getShapes():
    #generate common shapes found in traffic signs
    shapes = {
            'triangle' : generateRegularPolygonInt(3,-np.pi/6),
            'circle' : generateRegularPolygonInt(720),
            'square' : generateRegularPolygonInt(4,np.pi/4),
            'octagon' : generateRegularPolygonInt(8,np.pi/8),
            'rhombus' : generateRegularPolygonInt(4),
            }

    return shapes




def getShapePattern(coords,divisions):
    #coordinates are assumed to be ordered such that they complete a cyclic shape (i.e. triangle)

    n = len(coords)
    steps = int(np.ceil(divisions/n))

    sPattern = np.zeros(n*steps)
    j=0

    for i in range(n):
        diff = coords[(i+1)%n] - coords[i]
        for x in np.linspace(0,diff,num=steps):
            sPattern[j]=vecNorm(coords[i] + x)
            j+=1

    return sPattern[:divisions]


shapes = getShapes()

