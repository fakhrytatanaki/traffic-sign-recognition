import numpy as np
import cv2 as cv
import os
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



class SVMOneVsRest:
    #One Vs Rest Classifier 
    def loadDataset(self,datasetDirectory):

        self.models = []
        self.labelSet = [f for f in os.listdir(datasetDirectory) if not f.startswith('.')]

        print("[One-vs-Rest Classifier] loading datsets... ")

        for l in self.labelSet: #for every class
            svmModel = cv.ml.SVM_create()
            svmModel.setType(cv.ml.SVM_C_SVC)
            svmModel.setKernel(cv.ml.SVM_LINEAR)
            svmModel.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

            allData=[]
            allLabels=[]

            isClassData,isClassLabels = loadTrainingData(glob(os.path.join(datasetDirectory,l,'*.png')), 1)
            allData.append(isClassData)
            allLabels.append(isClassLabels)

            for nl in self.labelSet:
                if l!=nl:
                    print(os.path.join(datasetDirectory,nl,'*.png'))
                    isNotClassData,isNotClassLabel = loadTrainingData(glob(os.path.join(datasetDirectory,nl,'*.png')), -1) #anything other than class (l) is labeled as -1
                    allData.append(isNotClassData)
                    allLabels.append(isNotClassLabel)

            dataTrain = np.concatenate(allData)
            labelTrain = np.concatenate(allLabels)

            svmModel.train(dataTrain,cv.ml.ROW_SAMPLE, labelTrain)

            self.models.append(svmModel)

    def saveModels(self,dirname):
          for i,m in enumerate(self.models):
              m.save(os.path.join(dirname,f"{i}.{self.labelSet[i]}.xml"))




    def predict(self,imageSet):

        imageSetGrayScale = [cv.cvtColor(img,cv.COLOR_BGR2GRAY) for img in imageSet]
        modelReady = np.matrix([np.array(cv.resize(img,IMAGE_DATA_DIMENSIONS).flatten(),dtype=np.float32) for img in imageSetGrayScale],dtype=np.float32)

        modelOutputs = np.zeros((len(self.models),len(imageSet)))

        for i,m in enumerate(self.models):
            p = m.predict(modelReady)[1].flatten()
            modelOutputs[i,:] = p


        result = []
        
        for out in modelOutputs.T:
            result.append(self.labelSet[np.nanargmax(out)])

        return result





        

