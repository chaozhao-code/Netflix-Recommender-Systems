import numpy as np
from copy import deepcopy
import time
import warnings
import numpy as np
from copy import deepcopy
import time
import multiprocessing as mp
import argparse
import sys
import matplotlib.pylab as plt
warnings.filterwarnings("ignore")

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


class UVMatDecomp():
    def __init__(self, train, test, num_factors=10, num_iter=10, threshold=0.001):
        '''
        :param train: training data, size: M * N
        :param test: test data, size: M * N
        :param num_factors: number of factors
        '''
        self.train = train
        self.test = test
        self.d = num_factors  # number of factors
        self.num_iter = num_iter  # number of iteration
        self.global_average = np.nansum(self.train) / np.sum(~np.isnan(self.train))  # for initialization
        self.U = np.full((self.train.shape[0], self.d), np.sqrt(self.global_average / self.d))  # size: M * d
        self.V = np.full((self.d, self.train.shape[1]), np.sqrt(self.global_average / self.d))  # size: d * N

        # add noise
        np.random.seed(3645002)
        self.U += np.random.uniform(-1, 1, size=self.U.shape)
        self.V += np.random.uniform(-1, 1, size=self.V.shape)

        self.threshold = threshold

    def loss(self, pred, data='train'):
        '''
        :param pred: predicted value
        :param data: string variable, means calculate loss for 'train' dataset or 'test' dataset
        :return: RMSE, and MAE of the predicted values
        '''
        if data == 'train':
            real = self.train
        else:
            real = self.test
        MAE = np.nansum(np.absolute(real - pred)) / np.sum(~np.isnan(real))
        RMSE = np.sqrt(np.nansum((real - pred) ** 2) / np.sum(~np.isnan(real)))
        return RMSE, MAE

    def update(self):

        pred = self.U.dot(self.V)
        # print("Initial Train Loss: ", self.loss(pred, data='train'))
        # print("Initial Test Loss: ", self.loss(pred, data='test'))
        self.testRMSE = self.loss(pred, data='test')[0]

        for iter in range(self.num_iter):
            # update U
            for row in range(self.U.shape[0]):
                for col in range(self.U.shape[1]):
                    V_sj = self.V[col, :].reshape((1, -1))  # size: 1 * N
                    M_rj = self.train[row, :].reshape((1, -1))  # size: 1 * N

                    U_rk = np.delete(self.U[row, :].reshape((1, -1)), col, axis=1)  # size: 1 * (d-1)
                    V_kj = np.delete(self.V, col, axis=0)  # size: (d-1) * N
                    self.U[row, col] = np.nansum(V_sj * (M_rj - U_rk.dot(V_kj))) / np.nansum(
                        (~np.isnan(M_rj) * V_sj) ** 2)

            # update V
            for row in range(self.V.shape[0]):
                for col in range(self.V.shape[1]):
                    U_ir = self.U[:, row].reshape((-1, 1))  # size: M * 1
                    M_is = self.train[:, col].reshape((-1, 1))  # size: M * 1

                    U_ik = np.delete(self.U, row, axis=1)  # size: M * (d-1)
                    V_ks = np.delete(self.V[:, col], row, axis=0).reshape((-1, 1))  # size: (d-1) * 1

                    self.V[row, col] = np.nansum(U_ir * (M_is - U_ik.dot(V_ks))) / np.nansum(
                        (~np.isnan(M_is) * U_ir) ** 2)


            pred = self.U.dot(self.V)
            RMSE, MAE = self.loss(pred, data='test')

            if np.abs(RMSE - self.testRMSE) < 0.001:
                # if improvement less than 0.001, stop training
                break
            self.testRMSE = RMSE

        pred = self.U.dot(self.V)
        TRAIN_RMSE, TRAIN_MAE = self.loss(pred, data='train')
        TEST_RMSE, TEST_MAE = self.loss(pred, data='test')
        return TRAIN_RMSE, TRAIN_MAE, TEST_RMSE, TEST_MAE

    def update_plot(self, fig_path):

        pred = self.U.dot(self.V)
        # print("Initial Train Loss: ", self.loss(pred, data='train'))
        # print("Initial Test Loss: ", self.loss(pred, data='test'))
        self.testRMSE = self.loss(pred, data='test')[0]



        TRAIN_RMSE = []
        TRAIN_MAE = []
        TEST_RMSE = []
        TEST_MAE = []

        rmse, mae = self.loss(pred, data='train')
        TRAIN_RMSE.append(rmse)
        TRAIN_MAE.append(mae)
        rmse, mae = self.loss(pred, data='test')
        TEST_RMSE.append(rmse)
        TEST_MAE.append(mae)

        for iter in range(self.num_iter):
            # update U
            for row in range(self.U.shape[0]):
                for col in range(self.U.shape[1]):
                    V_sj = self.V[col, :].reshape((1, -1))  # size: 1 * N
                    M_rj = self.train[row, :].reshape((1, -1))  # size: 1 * N

                    U_rk = np.delete(self.U[row, :].reshape((1, -1)), col, axis=1)  # size: 1 * (d-1)
                    V_kj = np.delete(self.V, col, axis=0)  # size: (d-1) * N
                    self.U[row, col] = np.nansum(V_sj * (M_rj - U_rk.dot(V_kj))) / np.nansum(
                        (~np.isnan(M_rj) * V_sj) ** 2)

            # update V
            for row in range(self.V.shape[0]):
                for col in range(self.V.shape[1]):
                    U_ir = self.U[:, row].reshape((-1, 1))  # size: M * 1
                    M_is = self.train[:, col].reshape((-1, 1))  # size: M * 1

                    U_ik = np.delete(self.U, row, axis=1)  # size: M * (d-1)
                    V_ks = np.delete(self.V[:, col], row, axis=0).reshape((-1, 1))  # size: (d-1) * 1

                    self.V[row, col] = np.nansum(U_ir * (M_is - U_ik.dot(V_ks))) / np.nansum(
                        (~np.isnan(M_is) * U_ir) ** 2)


            pred = self.U.dot(self.V)

            rmse, mae = self.loss(pred, data='train')
            TRAIN_RMSE.append(rmse)
            TRAIN_MAE.append(mae)
            rmse, mae = self.loss(pred, data='test')
            TEST_RMSE.append(rmse)
            TEST_MAE.append(mae)

            if np.abs(rmse - self.testRMSE) < 0.001:
                # if improvement less than 0.001, stop training
                break
            self.testRMSE = rmse

        epoch = np.arange(1, len(TEST_RMSE)+1)
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
        plt.title('Loss vs. Epoch (UV Matrix Decomposition)')
        plt.legend()
        plt.savefig(fig_path)

def f(train_data, test_data, num_iter, num_factors, threshold):
    model = UVMatDecomp(train_data, test_data, num_iter=num_iter, num_factors=num_factors, threshold=threshold)
    return model.update()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_fold', type=int, default=5, help='Number of fold of Cross Validation')
    parser.add_argument('--num_factors', type=int, default=2, help='Number of factors of UV Matrix Decomposition')
    parser.add_argument('--num_iter', type=int, default=75, help='Number of iterations')
    parser.add_argument('--threshold', type=float, default=0.001, help='Threshold for stop training')
    parser.add_argument('--plot', action='store_true', help='Plot the loss of the training process without cv-fold')
    parser.add_argument('--fig_path', type=str, default='figure.png', help='Path for saving figure')
    opt = parser.parse_args()

    print(opt)


    start = time.time()
    cv = cv_split('ml-1m/ratings.dat')
    train, test = cv.split(n_fold=opt.cv_fold)
    end = time.time()
    print('Splite Dataset Success! Duration: ', end - start, 's.')

    if opt.plot:
        model = UVMatDecomp(train=train[0], test=test[0], num_factors=opt.num_factors, num_iter=opt.num_iter)
        model.update_plot(opt.fig_path)
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
            r.append(pool.apply_async(f, (train[i], test[i], opt.num_iter, opt.num_factors, opt.threshold)))


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

