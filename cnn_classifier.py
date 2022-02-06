import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import keras
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def to_one_hot(labels):
    one_hot_label = []
    for label in labels:
        temp = np.zeros(10)
        temp[label] = 1
        one_hot_label.append(temp)

    return np.asarray(one_hot_label)




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
trdata = trdata/255
trlabels = np.concatenate([Y1, Y2, Y3, Y4, Y5])

one_hot_trlabels = to_one_hot(trlabels)



#load test data
testdict = unpickle('/Users/Admin/Downloads/cifar-10-python/cifar-10-batches-py/test_batch')

testdata = testdict["data"]
testlabel = testdict["labels"]

testdata = np.array(testdata).astype("int32")
testdata = testdata/255
testlabel = np.array(testlabel)

one_hot_testlabel = to_one_hot(testlabel)


model = Sequential()
model.add(Dense(100, input_dim=3072, activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
keras.models.optimizer_v1.SGD(lr = 0.3)
model.compile(optimizer='sgd', loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(trdata,one_hot_trlabels, epochs=10, validation_data=(testdata,one_hot_testlabel))

train, =plt.plot(history.history['accuracy'],"-b" ,label='train_accuracy')
test, = plt.plot(history.history['val_accuracy'],"-g" ,label='test_accuracy')
plt.legend([train,test], ['train_accuracy', 'test_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.scatter(1,35.39,color='red')
plt.scatter(2,35.39,color='blue')
plt.scatter(3,36.6, color='green')
plt.scatter(4,40.80, color='yellow')
plt.legend(['1-NN', 'Naive bayes', 'bayes2x2', 'cnn'])
plt.ylabel('accuracy')
plt.xlabel('model')
plt.show()