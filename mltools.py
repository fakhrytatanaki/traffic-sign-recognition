import numpy as np
import cv2 as cv
import os
from glob import glob
from sys import getsizeof
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError,CategoricalCrossentropy 
from tensorflow.keras import layers

IMAGE_DATA_DIMENSIONS=(128,128)

def imageTransformer(img):
    return np.array(cv.resize(img,IMAGE_DATA_DIMENSIONS),dtype=np.float32)


def loadTrainingData(imageFiles,label):
    #loads a training data that belongs to one class

    trainingData = [cv.imread(f) for f in imageFiles]
    #trainingData =np.matrix([np.array(cv.resize(img,IMAGE_DATA_DIMENSIONS).flatten(),dtype=np.float32) for img in trainingData],dtype=np.float32)
    trainingData = np.array([ imageTransformer(img) for img in trainingData],dtype=np.float32)

    labels = np.zeros(len(trainingData),dtype=np.int32)
    labels[:] = label
    return trainingData,labels

def trainTestSplit(data,percentageTrain):
    n = len(data)
    nTrain = int(n*(percentageTrain/100))
    return np.copy(data[:nTrain]), np.copy(data[nTrain:n])



def shuffleData(data,labels,labelType=np.int32):
    dim = data.shape
    toShuffle = np.concatenate((data.reshape(len(labels),-1),labels.reshape(-1,1)),axis=1) 
    np.random.shuffle(toShuffle)
    data_shuffled = np.array(toShuffle[:,:-1],dtype=np.float32).reshape(dim)
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

class NeuralNetworkModel():
    def __init__(self,datasetDirectory):
        self.inputShape = (IMAGE_DATA_DIMENSIONS + (3,))
        self.labelSet = [f for f in os.listdir(datasetDirectory) if not f.startswith('.')]

        x_train = np.zeros(shape=((0,)+self.inputShape),dtype=np.float32)
        y_train = np.zeros(0,dtype=np.float32)

        for i,l in enumerate(self.labelSet):
            classData,classLabel = loadTrainingData(glob(os.path.join(datasetDirectory,l,'./*.png')), i)
            x_train = np.concatenate((x_train,classData))
            y_train = np.concatenate((y_train,classLabel))

        print(x_train.shape)
        print(y_train.shape)
        x_train,y_train = shuffleData(np.array(x_train), np.array(y_train))
        x_train = tf.keras.utils.normalize(x_train)

        # print(self.labelSet[y_train[30]])
        # plt.imshow(x_train[30].reshape(IMAGE_DATA_DIMENSIONS + (3,)))
        # plt.show()


        y_train_ohe = tf.one_hot(y_train,depth=len(self.labelSet))

        self.model = tf.keras.models.Sequential([
                layers.Conv2D(128, (3, 3), activation='relu', input_shape=self.inputShape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32,activation='sigmoid'),
                tf.keras.layers.Dense(16,activation='sigmoid'),
                tf.keras.layers.Dense(len(self.labelSet),activation='sigmoid'),
                ]
                )

        self.model.compile(
                optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy']
              )

        #print('ready dataset size : ',getsizeof(x_train))
        self.model.fit(x_train,y_train_ohe,epochs=50)

    def predict(self,imageSet):
        # imageSetGrayScale = [cv.cvtColor(img,cv.COLOR_BGR2GRAY) for img in imageSet]
        modelReady = np.array([imageTransformer(img) for img in imageSet],dtype=np.float32)
        modelReady = tf.keras.utils.normalize(modelReady)
        pred = self.model.predict(modelReady)
        return [self.labelSet[i] for i in np.argmax(pred,axis=1)]
        








class SVMOneVsRest:
    #One Vs Rest Classifier 
    def loadDataset(self,datasetDirectory):

        self.models = []
        self.labelSet = [f for f in os.listdir(datasetDirectory) if not f.startswith('.')]

        print("[One-vs-Rest Classifier] loading datsets... ")
        print("this might take a while....")

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
                    isNotClassData,isNotClassLabel = loadTrainingData(glob(os.path.join(datasetDirectory,nl,'*.png')), -1) #anything other than class (l) is labeled as -1
                    allData.append(isNotClassData)
                    allLabels.append(isNotClassLabel)

            dataTrain = np.concatenate(allData)
            labelTrain = np.concatenate(allLabels)

            svmModel.train(dataTrain,cv.ml.ROW_SAMPLE, labelTrain)

            self.models.append(svmModel)

            print(f"model for [ {l} ] is ready")

        print("all models are ready")

    def saveModels(self,dirname):
          for i,m in enumerate(self.models):
              m.save(os.path.join(dirname,f"{i}.{self.labelSet[i]}.xml"))




    def predict(self,imageSet):

        imageSetGrayScale = [cv.cvtColor(img,cv.COLOR_BGR2GRAY) for img in imageSet]
        modelReady = np.matrix([np.array(cv.resize(img,IMAGE_DATA_DIMENSIONS).flatten(),dtype=np.float32) for img in imageSetGrayScale],dtype=np.float32)

        modelOutputs = np.zeros((len(self.models),len(imageSet)))

        for i,m in enumerate(self.models):
            _p = m.predict(modelReady,cv.ml.StatModel_RAW_OUTPUT)
            p = _p[1].flatten()
            modelOutputs[i,:] = p


        result = []
        
        for out in modelOutputs.T:

            r = np.nanargmax(out)

            if out[r] < 0:
                result.append(None)
            else:
                result.append(self.labelSet[r])

        return result





        

