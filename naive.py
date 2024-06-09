#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import time
import tqdm
import warnings
warnings.filterwarnings("ignore")


# # Naive Approaches

# In[3]:


class cv_split():
    '''
    class for reading data from file path and split data for cross validation
    '''
    def __init__(self, path):
        '''
        :param path: string, file path of dataset
        '''
        self.path = path
        self.ratings = [] # list includes all rating data
        self.item_rating = np.full([6040, 3952], np.nan) # matrix, row represents user and column repersents movie


        with open(path) as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split('::')
                self.item_rating[int(line[0])-1, int(line[1])-1] = int(line[2])
                self.ratings.append([int(x) for x in line[0:3]])

        # rounding values bigger than 5 to 5 and smaller than 1 to 1
        for index, x in np.ndenumerate(self.item_rating):
            if ~np.isnan(x) and x < 1:
                self.item_rating[index[0], index[1]] = 1
            elif ~np.isnan(x) and x > 5:
                self.item_rating[index[0], index[1]] = 5

    def all(self):
        '''
        :return: do not split, return the whole dataset
        '''
        return self.item_rating

    def split(self, n_fold):
        '''
        :param n_fold: integer, fold of cross validation
        :return:
            train_set: dictionary, key is number, and values is training data for key experiment
            test_set: dictionary, key is number, and values is test data for key experiment
        '''
        train_set = {}
        test_set = {}
        np.random.seed(3645002)
        index = np.random.permutation(len(self.ratings))
        num_of_test = round(len(self.ratings)/5)
        for fold in range(n_fold):
            train_data = deepcopy(self.item_rating)
            test_data = np.full(train_data.shape, np.nan)

            if fold == 4:
                test_index = index[fold*num_of_test:]
            else:
                test_index = index[fold*num_of_test:(fold+1)*num_of_test]

            for i in test_index:
                t_index = self.ratings[i]
                test_data[t_index[0]-1, t_index[1]-1] = self.item_rating[t_index[0]-1, t_index[1]-1]
                train_data[t_index[0]-1, t_index[1]-1] = np.nan

            train_set[fold] = train_data
            test_set[fold] = test_data
        return train_set, test_set


# In[4]:


class naive_approaches():
    def __init__(self, train, test):
        '''
        :param train: training dataset, size: M * N, M means the number of users, N means the number of moives
        :param test: test dataset
        '''
        self.train = train
        self.test = test
        self.global_average = np.nansum(self.train) / np.sum(~np.isnan(self.train)) # for fall-back rules
        self.user_avg = np.zeros((self.train.shape[0],)) # size: M * 1, M means the number of users,
        self.moive_avg = np.zeros((self.train.shape[1],)) # size: N * 1, N means the number of moives

    def GlobalAverage(self):
        '''
        :return: predicted value of test dataset by global average
        '''
        glo_avg = np.nansum(self.train) / np.sum(~np.isnan(self.train))
        pred = np.full(self.test.shape, glo_avg)
        return pred

    def UserAverage(self):
        '''
        :return: predicted value of test dataset by user average
        '''
        user_avg = np.nanmean(self.train, axis=1)
        user_avg = np.nan_to_num(user_avg, nan=self.global_average) # fall-back rules

        pred = np.tile(user_avg, (self.train.shape[1], 1)).T
        return pred

    def MovieAverage(self):
        '''
        :return: predicted value of test dataset by moive average
        '''
        moive_avg = np.nanmean(self.train, axis=0)
        moive_avg = np.nan_to_num(moive_avg, nan=self.global_average)

        pred = np.tile(moive_avg, (self.train.shape[0], 1))
        return pred

    def LinearCombination(self, intercept=True):
        '''
        :param intercept: whether include the intercept parameter
        :return: predicted value of test dataset by linear regression
        '''

        # calculate user average
        self.user_avg = np.nanmean(self.train, axis=1)
        self.user_avg = np.nan_to_num(self.user_avg, nan=self.global_average)

        # calculate item average
        self.moive_avg = np.nanmean(self.train, axis=0)
        self.moive_avg = np.nan_to_num(self.moive_avg, nan=self.global_average)

        x = []
        y = []
        x_test = []
        for index, rate in np.ndenumerate(self.train):
            x_test.append([self.user_avg[index[0]], self.moive_avg[index[1]]])
            if ~np.isnan(rate):
                x.append([self.user_avg[index[0]], self.moive_avg[index[1]]])
                y.append(rate)
        x = np.array(x)
        y = np.array(y)
        x_test = np.array(x_test)
        reg = LinearRegression(fit_intercept=intercept)
        reg.fit(x, y)

        pred = reg.predict(x_test)
        pred = pred.reshape(self.train.shape)

        return pred



    def loss(self, pred, data='train'):
        '''
        :param pred: predicted value
        :param data: string variable, means calculate loss for 'train' dataset or 'test' dataset
        :return: RMSE, and MAE of the predicted values
        '''
        if data=='train':
            real = self.train
        else:
            real = self.test
        MAE = np.nansum(np.absolute(real-pred)) / np.sum(~np.isnan(real))
        RMSE = np.sqrt(np.nansum((real-pred)**2) / np.sum(~np.isnan(real)))
        return RMSE, MAE


# In[9]:


cv = cv_split('ml-1m/ratings.dat')
train, test = cv.split(n_fold=5)


# ## Global Average

# In[10]:


start = time.time()
TRAIN_MAE = 0
TRAIN_RMSE = 0
TEST_MAE = 0
TEST_RMSE = 0
for i in range(5):
    model = naive_approaches(train[i], test[i])
    pred = model.GlobalAverage()
    rmse, mae = model.loss(pred)
    TRAIN_RMSE += rmse
    TRAIN_MAE += mae
    rmse, mae = model.loss(pred, data='test')
    TEST_RMSE += rmse
    TEST_MAE += mae
TRAIN_MAE /= 5
TRAIN_RMSE /= 5
TEST_MAE /= 5
TEST_RMSE /= 5
end = time.time()
print("====Global Average====")
print("Running Time: ", end-start, 's.')
print("MAE of training data: ", TRAIN_MAE)
print("RMSE of training data: ", TRAIN_RMSE)
print("MAE of test data: ", TEST_MAE)
print("RMSE of test data: ", TEST_RMSE)


# ## User Average

# In[11]:


start = time.time()
TRAIN_MAE = 0
TRAIN_RMSE = 0
TEST_MAE = 0
TEST_RMSE = 0
for i in range(5):
    model = naive_approaches(train[i], test[i])
    pred = model.UserAverage()
    rmse, mae = model.loss(pred)
    TRAIN_RMSE += rmse
    TRAIN_MAE += mae
    rmse, mae = model.loss(pred, data='test')
    TEST_RMSE += rmse
    TEST_MAE += mae
TRAIN_MAE /= 5
TRAIN_RMSE /= 5
TEST_MAE /= 5
TEST_RMSE /= 5
end = time.time()
print("====User Average====")
print("Running Time: ", end-start, 's.')
print("MAE of training data: ", TRAIN_MAE)
print("RMSE of training data: ", TRAIN_RMSE)
print("MAE of test data: ", TEST_MAE)
print("RMSE of test data: ", TEST_RMSE)


# ## Movie Average

# In[12]:


start = time.time()
TRAIN_MAE = 0
TRAIN_RMSE = 0
TEST_MAE = 0
TEST_RMSE = 0
for i in range(5):
    model = naive_approaches(train[i], test[i])
    pred = model.MovieAverage()
    rmse, mae = model.loss(pred)
    TRAIN_RMSE += rmse
    TRAIN_MAE += mae
    rmse, mae = model.loss(pred, data='test')
    TEST_RMSE += rmse
    TEST_MAE += mae
TRAIN_MAE /= 5
TRAIN_RMSE /= 5
TEST_MAE /= 5
TEST_RMSE /= 5
end = time.time()
print("====Moive Average====")
print("Running Time: ", end-start, 's.')
print("MAE of training data: ", TRAIN_MAE)
print("RMSE of training data: ", TRAIN_RMSE)
print("MAE of test data: ", TEST_MAE)
print("RMSE of test data: ", TEST_RMSE)


# ## Linear Regression without Intercept

# In[13]:


start = time.time()
TRAIN_MAE = 0
TRAIN_RMSE = 0
TEST_MAE = 0
TEST_RMSE = 0
for i in range(5):
    model = naive_approaches(train[i], test[i])
    pred = model.LinearCombination(intercept=False)
    rmse, mae = model.loss(pred)
    TRAIN_RMSE += rmse
    TRAIN_MAE += mae
    rmse, mae = model.loss(pred, data='test')
    TEST_RMSE += rmse
    TEST_MAE += mae
TRAIN_MAE /= 5
TRAIN_RMSE /= 5
TEST_MAE /= 5
TEST_RMSE /= 5
end = time.time()
print("====Linear Regression without Intercept====")
print("Running Time: ", end-start, 's.')
print("MAE of training data: ", TRAIN_MAE)
print("RMSE of training data: ", TRAIN_RMSE)
print("MAE of test data: ", TEST_MAE)
print("RMSE of test data: ", TEST_RMSE)


# ## Linear Regression with Intercept

# In[ ]:


start = time.time()
TRAIN_MAE = 0
TRAIN_RMSE = 0
TEST_MAE = 0
TEST_RMSE = 0
for i in range(5):
    model = naive_approaches(train[i], test[i])
    pred = model.LinearCombination(intercept=True)
    rmse, mae = model.loss(pred)
    TRAIN_RMSE += rmse
    TRAIN_MAE += mae
    rmse, mae = model.loss(pred, data='test')
    TEST_RMSE += rmse
    TEST_MAE += mae
TRAIN_MAE /= 5
TRAIN_RMSE /= 5
TEST_MAE /= 5
TEST_RMSE /= 5
end = time.time()
print("====Linear Regression with Intercept====")
print("Running Time: ", end-start, 's.')
print("MAE of training data: ", TRAIN_MAE)
print("RMSE of training data: ", TRAIN_RMSE)
print("MAE of test data: ", TEST_MAE)
print("RMSE of test data: ", TEST_RMSE)


