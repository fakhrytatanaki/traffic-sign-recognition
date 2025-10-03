"""One-vs-rest SVM training and inference utilities built on OpenCV ML.

This module provides helpers to load image data into flattened grayscale
feature vectors and a simple `SVMOneVsRest` class that wraps a set of
binary linear SVMs (one per class) to enable multi-class classification.
"""
import numpy as np
import cv2 as cv
import os
from glob import glob
import matplotlib.pyplot as plt

IMAGE_DATA_DIMENSIONS=(512,512)

def loadTrainingData(imageFiles,label):
    """Load grayscale, resized images and return flattened samples with labels.

    Parameters:
        imageFiles: Iterable of file paths to input images.
        label: Integer label to assign to all images (e.g., 1 or -1).

    Returns:
        trainingData: numpy.matrix of shape (N, 512*512) with dtype float32
        labels: numpy.ndarray of shape (N,) with dtype int32
    """
    # loads a training data that belongs to one class
    trainingData = [cv.cvtColor(cv.imread(f),cv.COLOR_BGR2GRAY) for f in imageFiles]
    trainingData =np.matrix([np.array(cv.resize(img,IMAGE_DATA_DIMENSIONS).flatten(),dtype=np.float32) for img in trainingData],dtype=np.float32)
    labels = np.zeros(len(trainingData),dtype=np.int32)
    labels[:] = label
    return trainingData,labels

def trainTestSplit(data,percentageTrain):
    """Split an array into train and test portions by percentage.

    Parameters:
        data: Sequence or numpy array to split.
        percentageTrain: Percentage (0-100) to allocate to the train set.

    Returns:
        (train, test) copies of the data split.
    """
    n = len(data)
    nTrain = int(n*(percentageTrain/100))
    return np.copy(data[:nTrain]), np.copy(data[nTrain:n])



def shuffleData(data,labels,labelType=np.int32):
    """Randomly shuffle data and labels in unison and return shuffled copies.

    Parameters:
        data: 2D array-like of samples.
        labels: 1D array-like of integer labels.
        labelType: numpy dtype for returned labels.

    Returns:
        (data_shuffled, label_shuffled)
    """
    toShuffle = np.concatenate((data,labels.reshape(-1,1)),axis=1) 
    np.random.shuffle(toShuffle)
    data_shuffled = np.array(toShuffle[:,:-1],dtype=np.float32)
    label_shuffled = np.array(toShuffle[:,-1],dtype=labelType)
    return data_shuffled,label_shuffled

def calcAccuracy(yTest,yReal):
    """Compute accuracy in percent between predicted and true labels.

    Parameters:
        yTest: 1D array-like of predicted labels.
        yReal: 1D array-like of ground-truth labels.

    Returns:
        Accuracy as a float percentage in [0, 100].
    """
    assert(len(yTest)==len(yReal))

    correct = 0
    total = len(yReal)
    for i in range(total):
        if yTest[i]==yReal[i]:
            correct+=1


    return 100*correct/total



class SVMOneVsRest:
    """One-vs-rest multi-class classifier using a bank of linear SVMs."""
    def loadDataset(self,datasetDirectory):
        """Train one binary SVM per class using one-vs-rest labeling.

        Each subdirectory inside `datasetDirectory` is treated as a class name
        and should contain `.png` images. For a given class C, images from C
        are labeled +1 and all other class images are labeled -1.
        """

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
          """Save each trained SVM model as an XML file in `dirname`."""
          for i,m in enumerate(self.models):
              m.save(os.path.join(dirname,f"{i}.{self.labelSet[i]}.xml"))




    def predict(self,imageSet):
        """Predict labels for a list of BGR images.

        Parameters:
            imageSet: List of color images (BGR) as numpy arrays.

        Returns:
            List of predicted class names or None if all raw margins < 0.
        """

        imageSetGrayScale = [cv.cvtColor(img,cv.COLOR_BGR2GRAY) for img in imageSet]
        modelReady = np.matrix([np.array(cv.resize(img,IMAGE_DATA_DIMENSIONS).flatten(),dtype=np.float32) for img in imageSetGrayScale],dtype=np.float32)

        modelOutputs = np.zeros((len(self.models),len(imageSet)))

        for i,m in enumerate(self.models):
            # Use raw decision function outputs (margins) for one-vs-rest
            _p = m.predict(modelReady,cv.ml.StatModel_RAW_OUTPUT)
            p = _p[1].flatten()
            modelOutputs[i,:] = p


        result = []
        
        for out in modelOutputs.T:

            r = np.nanargmax(out)

            # If the highest margin is negative, reject as no confident class
            if out[r] < 0:
                result.append(None)
            else:
                result.append(self.labelSet[r])

        return result





        

