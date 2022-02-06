import pickle
import numpy as np
import skimage.transform as st
from scipy.stats import norm, multivariate_normal
import matplotlib as plt


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
    accuracy = round((match/tot_image)*100,2)

    return accuracy


def cifar10_color(X, image_length):
    resized_image = []
    rows = len(X)
    for image in X:
        for sub_image_index in range(image_length):
            new_row = np.mean(image[sub_image_index * 3072 // image_length: (sub_image_index + 1) * 3072 // image_length], axis=0,
                              dtype=np.float64)
            resized_image.append(new_row)
    resized_image = np.array(resized_image)
    resized_image = np.reshape(resized_image, (rows, image_length))
    return resized_image


def cifar_10_naivebayes_learn(Xp,Y):
    mu = []
    sigma = []
    p = []
    data_len = len(Xp)
    for class_num in range(10):
        class_member = []
        for index in range(data_len):
            if Y[index] == class_num:
                class_member.append(Xp[index])

        class_prior = len(class_member)/ data_len
        p.append(class_prior)
        class_array = np.asarray(class_member)
        class_mean = np.mean(class_array,axis= tuple(range(class_array.ndim-1)), dtype=np.float64)
        mu.append(class_mean)
        class_variance = np.std(class_array, axis= tuple(range(class_array.ndim-1)), dtype=np.float64)
        sigma.append(class_variance)

    mu_array = np.asarray(mu)
    sigma_array = np.asarray(sigma)
    p_array = np.asarray(p)

    return mu_array, sigma_array, p_array

def cifar_10_bayes_learn(Xf, Y):
    mu = []
    sigma = []
    p = []
    data_len = len(Xf)
    for class_num in range(10):
        class_member = []
        for index in range(data_len):
            if Y[index] == class_num:
                class_member.append(Xf[index])

        class_prior = len(class_member) / data_len
        p.append(class_prior)
        class_array = np.asarray(class_member)
        class_mean = np.mean(class_array, axis=0, dtype=np.float64)
        mu.append(class_mean)
        class_covariance = np.cov(class_array, rowvar= False , dtype=np.float64)
        sigma.append(class_covariance)

    mu_array = np.asarray(mu)
    sigma_array = np.asarray(sigma)
    p_array = np.asarray(p)

    return mu_array, sigma_array, p_array


def cifar10_classifier_bayes(X,mu,sigma,p):

    pred = []
    i = 0
    for data in X:
        probability = []
        class_prob_sum = 0
        for index in range(10):
            class_prob = multivariate_normal.pdf(data, mu[index], sigma[index]) * p[index]
            class_prob_sum += class_prob
            probability.append(class_prob)

        max_prob, prob_index = max((probability[i],i) for i in range(len(probability)))
        pred.append(prob_index)
        i = i + 1

    prediction = np.asarray(pred)

    return prediction

def cifar10_classifier_naivebayes(X,mu,sigma,p):

    pred = []
    i = 0
    for data in X:
        probability = []
        class_prob_sum = 0
        for index in range(10):
            p_red = (norm.pdf(data[0],mu[index][0], sigma[index][0]))
            p_blue = (norm.pdf(data[1],mu[index][1], sigma[index][1]))
            p_green = (norm.pdf(data[2], mu[index][2], sigma[index][2]))
            class_prob = p_red*p_blue*p_green* p[index]
            class_prob_sum += class_prob
            probability.append(class_prob)

        max_prob, prob_index = max((probability[i],i) for i in range(len(probability)))
        pred.append(prob_index)
        i = i + 1

    prediction = np.asarray(pred)

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



#task 1 naive bayesian classifier
#resize image to 1x1x3
#image_length = 1x1x3 = 3
resized_1x1_train = cifar10_color(trdata,3)
resized_1x1_test = cifar10_color(testdata,3)
mean_naive, variance_naive, prior_naive = cifar_10_naivebayes_learn(resized_1x1_train, trlabels)
pred_naive = cifar10_classifier_naivebayes(resized_1x1_test,mean_naive,variance_naive,prior_naive)
naive_bayes_classifier_accuracy = class_acc(pred_naive,testlabel)
print(f"the accuracy for the naive bayes classifier is {naive_bayes_classifier_accuracy}%")

#task 2 bayesian classifier
mean_bayes, covariance_bayes, prior_bayes = cifar_10_bayes_learn(resized_1x1_train,trlabels)
pred_bayes = cifar10_classifier_bayes(resized_1x1_test,mean_bayes,covariance_bayes,prior_bayes)
bayes_classifier_accuracy = class_acc(pred_bayes,testlabel)
print(f"the accuracy for the bayes classifier for 1x1 is {bayes_classifier_accuracy}%")

#task 3
#bayesian classifier with 2x2 resized image
#image length is = 2x2x3 =12
resized_2x2_train = cifar10_color(trdata,12)
resized_2x2_test = cifar10_color(testdata,12)
mean_2x2_bayes, covariance_2x2_bayes, prior_2x2_bayes = cifar_10_bayes_learn(resized_2x2_train,trlabels)
pred_2x2_bayes = cifar10_classifier_bayes(resized_2x2_test,mean_2x2_bayes,covariance_2x2_bayes,prior_2x2_bayes)
bayes_2x2_classifier_accuracy = class_acc(pred_2x2_bayes, testlabel)
print(f"the accuracy for the bayes classifier for 2x2 is {bayes_2x2_classifier_accuracy}%")

#bayesian classifier with 4x4 resized images
#image lenghth is = 4x4x3 = 48
resized_4x4_train = cifar10_color(trdata, 48)
resized_4x4_test = cifar10_color(testdata,48)
mean_4x4_bayes, covariance_4x4_bayes, prior_4x4_bayes = cifar_10_bayes_learn(resized_4x4_train,trlabels)
pred_4x4_bayes = cifar10_classifier_bayes(resized_4x4_test,mean_4x4_bayes,covariance_4x4_bayes,prior_4x4_bayes)
bayes_4x4_classifier_accuracy = class_acc(pred_4x4_bayes,testlabel)
print(f"the accuracy for the bayes classifier for 4x4 is {bayes_4x4_classifier_accuracy}%")

#bayesian classifier with 8x8 resized images
#image length is = 8x8x3 = 192
resized_8x8_train = cifar10_color(trdata,192)
resized_8x8_test = cifar10_color(testdata,192)
mean_8x8_bayes, covariance_8x8_bayes, prior_8x8_bayes = cifar_10_bayes_learn(resized_8x8_train, trlabels)
pred_8x8_bayes = cifar10_classifier_bayes(resized_8x8_test, mean_8x8_bayes, covariance_8x8_bayes, prior_8x8_bayes)
bayes_8x8_classifier_accuracy = class_acc(pred_8x8_bayes, testlabel)
print(f"the accuracy for the bayes classifier for 8x8 is {bayes_8x8_classifier_accuracy}%")

#bayesian classifier with 16x16 resized_image
#image length is = 16x16x3 = 768
resized_16x16_train = cifar10_color(trdata,768)
resized_16x16_test = cifar10_color(testdata,768)
mean_16x16_bayes, covariance_16x16_bayes, prior_16x16_bayes = cifar10_color(resized_16x16_train, trlabels)
pred_16x16_bayes = cifar10_classifier_bayes(resized_16x16_test, mean_16x16_bayes, covariance_16x16_bayes,prior_16x16_bayes)
bayes_16x16_classifier_accuracy = class_acc(pred_16x16_bayes,testlabel)
print(f"the accuracy for the bayes classifier for 16x16 is {bayes_16x16_classifier_accuracy}%")


