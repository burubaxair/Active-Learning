# A script that uses Gaussian Process Regressor 
# for the exercise at page 18, Figure 2.6. 
# of the book "Active Learning" by Burr Settles, 
# Link to the book: http://active-learning.net/

# See also https://stackoverflow.com/questions/50171114/gaussian-process-regression-sudden-increase-of-the-predictions-variance

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.random.seed(1)


x = np.atleast_2d(np.linspace(-10,10,200)).T

mu = 0
sig = 2

def gaussian(x, mu, sig):
    return np.exp(-np.square((x-mu)/sig)/2)

x_train = np.atleast_2d(sig * np.random.randn(1,2) + mu).T
#training data

def fit_GP(x_train):

    y_train = gaussian(x_train, mu, sig).ravel()

    # Instanciate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(x_train, y_train)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    return y_train, y_pred, sigma

def plot_GP(n_iter):

    f, ax = plt.subplots(2, sharex=True)
    ax[0].set_title('Iteration '+str(n_iter+1))
    ax[0].plot(x, gaussian(x, mu, sig), color="red", label="ground truth")
    ax[0].scatter(x_train, y_train, color='navy', s=30, marker='o', label="training data")
    ax[0].plot(x, y_pred, 'b-', color="blue", label="prediction")
    ax[0].fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% conf. intvl')
    ax[0].legend(loc='best')
    ax[1].plot(x,sigma)
    plt.savefig('gp'+str(n_iter+1)+'.png')
    plt.show()


n_iter = 15 # number of iterations

for i in range(n_iter):
    y_train, y_pred, sigma = fit_GP(x_train)
    plot_GP(i)
    x_train = np.vstack((x_train, x[np.argmax(sigma)]))
