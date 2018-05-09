# A script that uses RandomForestClassifier and a pool-based scenario 
# for the exercise at page 20, Figure 2.8. 
# of the book "Active Learning" by Burr Settles, 
# Link to the book: http://active-learning.net/

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(1)

low1 = -1; high1 = 1; low2 = -2; high2 = 2;

n_p = 40

range1 = np.linspace(low1, high1,n_p/2, endpoint=False)
range2 = np.linspace(low2, high2,n_p, endpoint=False)

cl1 = np.dstack(np.meshgrid(range1, range1)).reshape(-1, 2)
cl = np.dstack(np.meshgrid(range2, range2)).reshape(-1, 2)       # outer rectangle
cl2 = cl[(np.abs(cl + 0.1) >= 1).any(1)]                 # take out the inner rectangle

n_points1 = len(cl1)
n_points2 = len(cl2)

y1 = np.zeros((n_points1,1))
y2 = np.ones((n_points2,1))

X = np.vstack((cl1,cl2))
y = np.vstack((y1,y2))

y[np.absolute(X[:,0]-X[:,1])<0.15]=1

X_colors = ["red" if i else "green" for i in y]
plt.scatter(X[:,0], X[:,1], s = 20, marker='s', c=X_colors)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9875)

reg = RandomForestClassifier()

n_iter = 20 # number of iterations

for i in range(n_iter):

    reg.fit(X_train, y_train.ravel())
    yp = reg.predict(X_test)
    pred = reg.predict_proba(X_test)
    ind_min_conf = np.argmin(np.absolute(pred-0.5),axis=0)[0]
    print('Least confident: ',X_test[ind_min_conf,:],y_test[ind_min_conf],'\n')

    train_colors = ["red" if i else "green" for i in y_train]
    test_colors = ["red" if i else "green" for i in yp]

    plt.scatter(X[:,0], X[:,1], s = 3, marker='s', c=X_colors)
    plt.scatter(X_train[:,0], X_train[:,1], s = 30, marker='s', c=train_colors)
    plt.scatter(X_test[:,0],X_test[:,1], s = 30, marker='s', c=test_colors,alpha=0.1)

    plt.xlim((-2, 2))   
    plt.ylim((-2, 2))
    plt.title('Iteration '+str(i+1))
    plt.grid()
    plt.show()

    X_train = np.vstack((X_train,X_test[ind_min_conf,:]))
    y_train = np.vstack((y_train,y_test[ind_min_conf,:]))

    X_test = X_test[[i for i in range(len(X_test)) if i != ind_min_conf],:]
    y_test = y_test[[i for i in range(len(y_test)) if i != ind_min_conf],:]
