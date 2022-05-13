import numpy as np
import cv2 as cv
from glob import glob

def loadTrainingData(imageFiles,label):
    trainingData = [cv.cvtColor(cv.imread(f),cv.COLOR_BGR2GRAY) for f in imageFiles]
    trainingData =np.matrix([np.array(cv.resize(img,(512,512)).flatten(),dtype=np.float32) for img in trainingData],dtype=np.float32)
    labels = np.zeros(len(trainingData),dtype=np.int32)
    labels[:] = label
    return trainingData,labels

def trainTestSplit(data,percentageTrain):
    n = len(data)
    nTrain = int(n*(percentageTrain/100))
    return np.copy(data[:nTrain]), np.copy(data[nTrain:n])





stopData,stopLabels = loadTrainingData(glob('./dataset/stop/*.png'),1) 
notStopData,notStopLabels = loadTrainingData(glob('./dataset/[!stop]*/*.png'),-1) #anything other than stop

data = np.concatenate((stopData,notStopData))
labels = np.concatenate((stopLabels,notStopLabels))


dataTrain,dataTest = trainTestSplit(data, 80)
labelTrain,labelTest = trainTestSplit(labels, 80)


svmModel = cv.ml.SVM_create()

svmModel.setType(cv.ml.SVM_C_SVC)
svmModel.setKernel(cv.ml.SVM_LINEAR)
svmModel.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

svmModel.train(dataTrain,cv.ml.ROW_SAMPLE, labelTrain)


