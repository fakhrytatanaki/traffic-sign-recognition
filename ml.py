import numpy as np
import cv2 as cv
from glob import glob

def loadTrainingData(imageFiles,label):
    trainingData = [cv.cvtColor(f,cv.COLOR_BGR2GRAY) for f in files]
    trainingData = np.array([cv.resize(img,(512,512)) for img in trainingData],dtype=np.float32)
    labels = np.zeros(len(trainingData),dtype=np.float32)
    labels[:] = label
    return trainingData,labels

def trainTestSplit(data,perc):





svmModel = cv.ml.SVM_create()

svmModel.setType(cv.ml.SVM_C_SVC)
svmModel.setKernel(cv.ml.SVM_LINEAR)
svmModel.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

loadTrainingData(glob('./dataset/stop/*.png'))




