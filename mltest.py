import numpy as np
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt

IMAGE_DATA_DIMENSIONS=(512,512)

def loadTrainingData(imageFiles,label):
    #loads a training data that belongs to one class

    trainingData = [cv.cvtColor(cv.imread(f),cv.COLOR_BGR2GRAY) for f in imageFiles]

    trainingData =np.matrix([np.array(cv.resize(img,IMAGE_DATA_DIMENSIONS).flatten(),dtype=np.float32) for img in trainingData],dtype=np.float32)
    labels = np.zeros(len(trainingData),dtype=np.int32)
    labels[:] = label
    return trainingData,labels

def trainTestSplit(data,percentageTrain):
    n = len(data)
    nTrain = int(n*(percentageTrain/100))
    return np.copy(data[:nTrain]), np.copy(data[nTrain:n])



def shuffleData(data,labels,labelType=np.int32):
    toShuffle = np.concatenate((data,labels.reshape(-1,1)),axis=1) 
    np.random.shuffle(toShuffle)
    data_shuffled = np.array(toShuffle[:,:-1],dtype=np.float32)
    label_shuffled = np.array(toShuffle[:,-1],dtype=labelType)
    return data_shuffled,label_shuffled

def calcAccuracy(yTest,yReal):
    assert(len(yTest)==len(yReal))

    correct = 0
    total = len(yReal)
    for i in range(total):
        if yTest[i]==yReal[i]:
            correct+=1


    return 100*correct/total





stopData,stopLabels = loadTrainingData(glob('./dataset/stop/*.png'),1) 
notStopData,notStopLabels = loadTrainingData(glob('./dataset/[!stop]*/*.png'),-1) #anything other than stop

data = np.concatenate((stopData,notStopData))
labels = np.concatenate((stopLabels,notStopLabels))

data,labels = shuffleData(data, labels)

dataTrain,dataTest = trainTestSplit(data, 30)
labelTrain,labelTest = trainTestSplit(labels, 30)


svmModel = cv.ml.SVM_create()

svmModel.setType(cv.ml.SVM_C_SVC)
svmModel.setKernel(cv.ml.SVM_LINEAR)
svmModel.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

svmModel.train(dataTrain,cv.ml.ROW_SAMPLE, labelTrain)




ret,result = svmModel.predict(dataTest)
print(calcAccuracy(result, labelTest))






# for d,i in enumerate(dataTest):
#     y = svmModel.predict(d)[1]
#     print(labelTest[i],y)
