import pickle
import numpy as np
import random
from scipy import stats

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict



def class_acc(pred, gt):
    match = 0
    tot_image = len(pred)
    for index in range(tot_image):
        if pred[index] == gt[index]:
            match+=1
    accuracy = (match/tot_image)*100

    return f"Model has accuracy of {round(accuracy,2)}%"

def cifar_10_classifier_random(x):

    pred = []
    for index in range(len(x)):
        pred.append(random.randint(0,9))

    return pred

def euclid_dist(test, train):

    test_sqr_sum = np.sum(test**2)
    train_sqr_sum = np.sum(train**2)
    dot_prod = np.dot(test,train)
    dist = np.sqrt(test_sqr_sum + train_sqr_sum - 2*dot_prod)
    return dist

def cifar_10_classifier_1nn(x, trdata,trlabels):
    prediction = []
    for index in range(len(x)):
        distance = []
        for j in range(len(trdata)):
            #dist = np.linalg.norm(np.array(x[index]) - np.array(trdata[j]))
            dist = euclid_dist(np.array(x[index]), np.array(trdata[j]))
            distance.append(dist)

        nearest_image_dist = min(distance)
        image_pos = distance.index(nearest_image_dist)
        prediction.append(trlabels[image_pos])
    return prediction

def cifar_10_classifier_knn(x, trdata, trlabels):
    prediction = []
    for index in range(len(x)):
        distance = []
        for train in range(len(trdata)):
            dist = euclid_dist(np.array(x[index]), np.array(trdata[train]))
            distance.append(dist)
        dist_copy = distance[:]
        distance.sort()
        nearest_k_neigbours = distance[:3]
        image_pos = []
        for i in nearest_k_neigbours:
            pos = dist_copy.index(i)
            image_pos.append(trlabels[pos])

        pred = stats.mode(image_pos)
        prediction.append(pred[0].item())
    return prediction

datadict1 = unpickle('/Users/Admin/Downloads/cifar-10-python/cifar-10-batches-py/data_batch_1')
datadict2 = unpickle('/Users/Admin/Downloads/cifar-10-python/cifar-10-batches-py/data_batch_2')
datadict3 = unpickle('/Users/Admin/Downloads/cifar-10-python/cifar-10-batches-py/data_batch_3')
datadict4 = unpickle('/Users/Admin/Downloads/cifar-10-python/cifar-10-batches-py/data_batch_4')
datadict5 = unpickle('/Users/Admin/Downloads/cifar-10-python/cifar-10-batches-py/data_batch_5')


X1 = datadict1["data"]
Y1 = datadict1["labels"]

X2 = datadict2["data"]
Y2 = datadict2["labels"]

X3 = datadict3["data"]
Y3 = datadict3["labels"]

X4 = datadict4["data"]
Y4 = datadict4["labels"]

X5 = datadict5["data"]
Y5 = datadict5["labels"]


trdata = np.concatenate([X1, X2, X3, X4, X5]).astype("int32")
trlabels = np.concatenate([Y1, Y2, Y3, Y4, Y5])


#load test data
testdict = unpickle('/Users/Admin/Downloads/cifar-10-python/cifar-10-batches-py/test_batch')

testdata = testdict["data"]
testlabel = testdict["labels"]

testdata = np.array(testdata).astype("int32")
testlabel = np.array(testlabel)


predict = cifar_10_classifier_knn(testdata, trdata, trlabels)
print("3NN CLASSIFIER")
print(predict)
print(class_acc(predict, testlabel))



