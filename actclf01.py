# A script that uses Logistic Regression 
# for the exercise at page 13, Figure 2.2. 
# of the book "Active Learning" by Burr Settles, 
# Link to the book: http://active-learning.net/

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

np.random.seed(1)

mu1 = np.array([-2,0])
mu2 = np.array([2,0])

sig = 1

n_points = 200

cl1 = sig * np.random.randn(n_points,2) + mu1
cl2 = sig * np.random.randn(n_points,2) + mu2

y1 = np.zeros((n_points,1))
y2 = np.ones((n_points,1))

X = np.vstack((cl1,cl2))
y = np.vstack((y1,y2))

ind1 =  np.random.choice(range(n_points), size=1, replace=False)
ind2 =  np.random.choice(range(n_points), size=1, replace=False)

X1_train, X1 = cl1[ind1,:], cl1[[i for i in range(n_points) if i not in ind1],:]
y1_train, y1 = y1[ind1,:], y1[[i for i in range(n_points) if i not in ind1],:]
X2_train, X2 = cl2[ind2,:], cl2[[i for i in range(n_points) if i not in ind2],:]
y2_train, y2 = y2[ind2,:], y2[[i for i in range(n_points) if i not in ind2],:]

X = np.vstack((X1,X2))
y = np.vstack((y1,y2))

X_train = np.vstack((X1_train,X2_train))
y_train = np.vstack((y1_train,y2_train))

logreg = LogisticRegression()

n_iter = 15 # number of iterations

for i in range(n_iter):

    logreg.fit(X_train, y_train.ravel())
    pred = logreg.predict_proba(X)
    ind_min_conf = np.argmin(np.absolute(pred-0.5),axis=0)[0]
    print('Least confident: ',X[ind_min_conf,:],y[ind_min_conf],'\n')

    w = logreg.coef_[0]
    a = -w[0] / w[1]

    xx = np.linspace(-4,4)
    yy = a * xx - (logreg.intercept_[0]) / w[1]

    class_colors = ["green", "red"]
    plt.scatter(cl1[:,0], cl1[:,1], s = 2, marker='.', color='green')
    plt.scatter(cl2[:,0], cl2[:,1], s = 2, marker='.', color='red')
    plt.scatter(X_train[:,0], X_train[:,1], s = 50, marker='s', c=class_colors)
    plt.plot(xx, yy, 'b-')
    plt.xlim((-5, 5))   
    plt.ylim((-5, 5))
    plt.title('Iteration '+str(i+1))
    plt.grid()
    plt.show()

    X_train = np.vstack((X_train,X[ind_min_conf,:]))
    y_train = np.vstack((y_train,y[ind_min_conf,:]))

    X = X[[i for i in range(len(X)) if i != ind_min_conf],:]
    y = y[[i for i in range(len(y)) if i != ind_min_conf],:]

