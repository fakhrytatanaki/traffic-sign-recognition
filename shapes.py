"""Shape utilities for generating canonical traffic sign contours.

Provides helpers to generate regular polygons (triangle, square, octagon)
and a dense circle approximation, as well as utilities to subdivide edges
and compute radial distance patterns for contour matching.
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import json

vecNorm = lambda x,y=0:np.sqrt(np.sum((x-y)**2))

def subdivide(polygonPoints,divisions):
    """Subdivide each edge of a polygon into equal segments.

    Parameters:
    	polygonPoints: ndarray of shape (N, 2) of ordered polygon vertices.
    	divisions: Number of segments per polygon edge.

    Returns:
    	ndarray of shape (N*divisions, 2) with interpolated points.
    """
    n=polygonPoints.shape[0]
    res = np.zeros(shape=(n*divisions,polygonPoints.shape[1]),dtype=polygonPoints.dtype)

    for i in range(n):
        diff = polygonPoints[(i+1)%n]-polygonPoints[i]
        for j in range(divisions):
            res[i*divisions + j] = polygonPoints[i] + (diff/divisions)*j

    return res
    
def generateRegularPolygonInt(nsides,angleOffset=0,r=300):
    """Generate integer coordinates for a regular polygon.

    Parameters:
    	nsides: Number of polygon sides.
    	angleOffset: Rotation offset in radians.
    	r: Radius used to scale coordinates before integer casting.

    Returns:
    	ndarray of shape (nsides+1, 2) closing the polygon.
    """

    coords = np.zeros((nsides+1,2),dtype=np.int32)

    for i,theta in enumerate(np.linspace(0,2*np.pi,num=nsides+1)):
            coords[i] = [int(r*np.cos(angleOffset + theta)),int(r*np.sin(angleOffset + theta))]

    return coords;

def generateRegularPolygon(nsides,angleOffset=0):
    """Generate float coordinates for a regular polygon on the unit circle.

    Parameters:
    	nsides: Number of polygon sides.
    	angleOffset: Rotation offset in radians.

    Returns:
    	ndarray of shape (nsides+1, 2) closing the polygon.
    """

    coords = np.zeros((nsides+1,2))

    for i,theta in enumerate(np.linspace(0,2*np.pi,num=nsides+1)):
            coords[i] = [np.cos(angleOffset + theta),np.sin(angleOffset + theta)]

    return coords;

def getShapes():
    """Return canonical contour templates for common traffic sign shapes."""
    shapes = {
            'triangle' : generateRegularPolygonInt(3,-np.pi/6),
            'circle' : generateRegularPolygonInt(720),
            'square' : generateRegularPolygonInt(4,np.pi/4),
            'octagon' : subdivide(generateRegularPolygonInt(8,np.pi/8),10)
            }

    return shapes




def getShapePattern(coords,divisions):
    """Compute radial-distance signature sampled along a closed contour.

    Assumes `coords` are ordered to complete a cyclic shape (e.g., triangle).
    """

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




