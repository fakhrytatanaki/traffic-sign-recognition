import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json

# TODO : add useful documentations/comments

def generateSignature(imgarr,n,thresh):

    signature = np.zeros(n)

    test = np.zeros(shape=imgarr.shape)
    step = 3
    for i,theta in enumerate(np.linspace(0,2*np.pi,num=n)):
        ycenter,xcenter = imgarr.shape[0]//2,imgarr.shape[1]//2
        x = 0
        y = 0
        xstep = step*np.cos(theta)
        ystep = step*np.sin(theta)

        diff=0
        while 0 <= ycenter + int(y) < imgarr.shape[0] - np.ceil(ystep) and 0 <= xcenter + int(x) < imgarr.shape[1] - np.ceil(xstep) and diff < thresh:
            diff=np.abs(imgarr[ycenter + int(y+ystep),xcenter + int(x + xstep)] - imgarr[ycenter + int(y-ystep),xcenter + int(x-xstep)])
            x+=xstep
            y+=ystep
        test[ycenter + int(y-ystep),xcenter + int(x-xstep)]=255

        signature[i]=np.sqrt(x**2 + y**2)
    return signature/(xcenter)

def generateSignatureModels(dirs,n,thresh,labels=None):

    if not labels:
        labels = range(0,len(dirs))

    data = {}
    data['vecSize']=n
    data['sigModels']={}
    
    for i,d in enumerate(dirs):
        data['sigModels'][labels[i]]=generateSignature(plt.imread(d), n, thresh)
    return sigModels

def saveSignatureModels(fname,data):
    

    serializableData = {}
    serializableData['vecSize']=data['vecSize']
    serializableData['sigModels']={}

    for k in data['sigModels']:
        serializableData['sigModels'][k] = list(data['sigModels'][k]) #numpy arrays has to be converted to lists before saving as json

    with open(fname,'w+') as fp:
        json.dump(serializableData,fp)

def loadSignatureModels(fname):
    with open(fname,'r') as fp:
       sigModels = json.load(fp)

    for k in sigModels:
        sigModels[k] = np.array(sigModels[k])
    return sigModels
    




class SignatureModel:
    def __init__(self,fname):
        data = loadSignatureModels(fname)
        self.vecSize = data['vecSize']
        self.models = data['sigModels']

# TODO : finish the signature prediction function
    def predict(self,image,boundaries,errThresh,diffThresh):

        for b in boundaries:
            for sig in self.models:
                boundarySig = generateSignature(image[b[0][0]:b[1][0],b[0][1]:b[1][1]], n, thresh)












    
    

