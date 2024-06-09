#!/usr/bin/env python
# coding: utf-8



import numpy as np
from copy import deepcopy
import time
import multiprocessing as mp
import argparse
import sys
import matplotlib.pylab as plt

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


class MatrixFactorization():
    def __init__(self, train_data, test_data, learn_rate=0.005, reg=0.05, num_factors=10, num_iter=75, fig_path=None):
        self.train_data = train_data # size: M * N, M means the number of users, N means the number of moives
        self.test_data = test_data # size: M * N
        self.learn_rate = learn_rate # learning rate
        self.reg = reg # regularization rate
        self.num_factor = num_factors # number of features for moives and users
        self.num_iter = num_iter # number of iteration
        self.fig_path = fig_path # path for saving plot
        np.random.seed(3644677)
        self.U = np.random.normal(size=(train_data.shape[0], self.num_factor)) # size: M * F, F means the number of factors
        self.V = np.random.normal(size=(self.num_factor, train_data.shape[1])) # size: F * N

    def loss(self, pred, real):
        '''
        :param pred: predicted value
        :param real: real value
        :return: RMSE, and MAE of the predicted values
        '''
        MAE = np.nansum(np.absolute(real-pred)) / np.sum(~np.isnan(real))
        RMSE = np.sqrt(np.nansum((real-pred)**2) / np.sum(~np.isnan(real)))
        return RMSE, MAE

    def SGD(self):
        '''
        :return: RMSE, MAR of the predicted values for training data and test data
        '''
        # start = time.time()
        for i in range(self.num_iter):
            for index, x in np.ndenumerate(self.train_data):
                if ~np.isnan(x):
                    pred = self.U[index[0], :].dot(self.V[:, index[1]])
                    error = x - pred
                    self.U[index[0], :] = self.U[index[0], :] + self.learn_rate * (error * self.V[:, index[1]] - self.reg * self.U[index[0], :])
                    self.V[:, index[1]] = self.V[:, index[1]] + self.learn_rate * (error * self.U[index[0], :] - self.reg * self.V[:, index[1]])
        # end = time.time()

        # print('Running Time: ', end - start, 's.')
        pred = np.dot(self.U, self.V)
        TRAIN_RMSE, TRAIN_MAE = self.loss(pred, self.train_data)
        TEST_RMSE, TEST_MAE = self.loss(pred, self.test_data)
        return TRAIN_RMSE, TRAIN_MAE, TEST_RMSE, TEST_MAE

    def plotSGD(self):
        '''
        Plot the Loss vs. Epoch and save the fig
        :return:
        '''
        TRAIN_RMSE = []
        TRAIN_MAE = []
        TEST_RMSE = []
        TEST_MAE = []
        epoch = np.arange(1, self.num_iter+1)
        for i in range(self.num_iter):
            for index, x in np.ndenumerate(self.train_data):
                if ~np.isnan(x):
                    pred = self.U[index[0], :].dot(self.V[:, index[1]])
                    error = x - pred
                    self.U[index[0], :] = self.U[index[0], :] + self.learn_rate * (error * self.V[:, index[1]] - self.reg * self.U[index[0], :])
                    self.V[:, index[1]] = self.V[:, index[1]] + self.learn_rate * (error * self.U[index[0], :] - self.reg * self.V[:, index[1]])
            pred = np.dot(self.U, self.V)
            rmse, mae = self.loss(pred, self.train_data)
            TRAIN_RMSE.append(rmse)
            TRAIN_MAE.append(mae)
            rmse, mae = self.loss(pred, self.test_data)
            TEST_RMSE.append(rmse)
            TEST_MAE.append(mae)
        TRAIN_RMSE = np.array(TRAIN_RMSE)
        TRAIN_MAE = np.array(TRAIN_MAE)
        TEST_RMSE = np.array(TEST_RMSE)
        TEST_MAE = np.array(TEST_MAE)
        plt.plot(epoch, TRAIN_RMSE, label='RMSE Training Data')
        plt.plot(epoch, TRAIN_MAE, label='MAE Training Data')
        plt.plot(epoch, TEST_RMSE, label='RMSE Test Data')
        plt.plot(epoch, TEST_MAE, label='MAE Test Data')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()
        plt.savefig(self.fig_path)


    def save(self, path1, path2):
        '''
        Save U and V into files
        :param path1: path used to save U
        :param path2: path used to save V
        :return:
        '''
        with open(path1, 'wb') as f:
            np.save(f, self.U)
        with open(path2, 'wb') as f:
            np.save(f, self.V)


def f(train_data, test_data, num_iter, num_factors, learn_rate, regularization):
    model = MatrixFactorization(train_data, test_data, num_iter=num_iter, num_factors=num_factors, learn_rate=learn_rate, reg=regularization)
    return model.SGD()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--whole', action='store_true', help='Using the whole dataset as train dataset and save files for part 2')
    parser.add_argument('--cv_fold', type=int, default=5, help='Number of fold of Cross Validation')
    parser.add_argument('--num_factors', type=int, default=10, help='Number of factors of Matrix Factorization')
    parser.add_argument('--num_iter', type=int, default=75, help='Number of iterations')
    parser.add_argument('--regularization', type=float, default=0.05, help='Regularization coefficient')
    parser.add_argument('--learn_rate', type=float, default=0.005, help='Learning Rate')
    parser.add_argument('--plot', action='store_true', help='Plot the loss of the training process without cv-fold')
    parser.add_argument('--fig_path', type=str, default='figure.png', help='Path for saving figure')
    opt = parser.parse_args()

    print(opt)


    start = time.time()
    cv = cv_split('ml-1m/ratings.dat')
    train, test = cv.split(n_fold=opt.cv_fold)
    end = time.time()
    print('Splite Dataset Success! Duration: ', end - start, 's.')

    if opt.whole:
        all_data = cv.all()
        model = MatrixFactorization(train_data=all_data, test_data=all_data, learn_rate=opt.learn_rate, reg=opt.regularization, num_factors=opt.num_factors, num_iter=opt.num_iter)
        model.SGD()
        model.save('User.npy', 'Item.npy')
        sys.exit(0)

    if opt.plot:
        model = MatrixFactorization(train_data=train[0], test_data=test[0], learn_rate=opt.learn_rate,
                                    reg=opt.regularization, num_factors=opt.num_factors, num_iter=opt.num_iter, fig_path=opt.fig_path)
        model.plotSGD()
        print('Finished! The path of the figure: ', opt.fig_path)
        sys.exit(0)




    start = time.time()
    train_RMSE = 0
    train_MAE = 0
    test_RMSE = 0
    test_MAE = 0
    with mp.Pool(processes=opt.cv_fold) as pool:
        r = []
        for i in range(opt.cv_fold):
            r.append(pool.apply_async(f, (train[i], test[i], opt.num_iter, opt.num_factors, opt.learn_rate, opt.regularization)))


        # for i in range(opt.cv_fold):
        #     r[i].join()
        for i in range(opt.cv_fold):
            result = r[i].get()
            train_RMSE += result[0]
            train_MAE += result[1]
            test_RMSE += result[2]
            test_MAE += result[3]
    train_RMSE /= opt.cv_fold
    train_MAE /= opt.cv_fold
    test_RMSE /= opt.cv_fold
    test_MAE /= opt.cv_fold


    end = time.time()
    print(f'Average Train RMSE {train_RMSE}')
    print(f'Average Train MAE {train_MAE}')
    print(f'Average TEST RMSE {test_RMSE}')
    print(f'Average TEST MAE {test_MAE}')
    print("Running time: ", end - start, 's.')

 