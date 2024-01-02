import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 

class ImageClassifier:
    def __init__(self, input_dir='./clf-data', categories=['empty', 'not_empty']):
        self.input_dir = input_dir
        self.categories = categories
        self.data = None
        self.labels = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.best_estimator = None

    def getDataAndLabels(self):
        data = []
        labels = []

        for idx, category in enumerate(self.categories):
            for file in os.listdir(os.path.join(self.input_dir, category)):
                img_path = os.path.join(self.input_dir, category, file)
                img = imread(img_path)
                img = resize(img, (15, 15))
                data.append(img.flatten())
                labels.append(idx)
        self.data = np.asarray(data)
        self.labels = np.asarray(labels)

    def getTrainTestSplit(self):
        # train / test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=0.2, shuffle=True, stratify=self.labels
        )

    def getBestEstimator(self):
        parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
        classfier = SVC()
        grid_search = GridSearchCV(classfier, parameters)
        grid_search.fit(self.x_train, self.y_train)
        self.best_estimator = grid_search.best_estimator_

    def modelTraining(self):
        self.getDataAndLabels()
        self.getTrainTestSplit()
        self.getBestEstimator()

    def modelPrediction(self):
        y_prediction = self.best_estimator.predict(self.x_test)
        score = accuracy_score(y_prediction, self.y_test)
        print('{}% of samples were correctly classified'.format(str(score * 100)))

    def saveModel(self, filename='./model.p'):
        pickle.dump(self.best_estimator, open(filename, 'wb'))


if __name__ == "__main__":
    classifier = ImageClassifier()
    classifier.modelTraining()
    classifier.modelPrediction()
    classifier.saveModel()
