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

ind1 =  np.random.choice(range(n_points), size=1, replace=False)
ind2 =  np.random.choice(range(n_points), size=1, replace=False)

X1_train, X1 = cl1[ind1,:], cl1[[i for i in range(n_points) if i not in ind1],:]
y1_train, y1 = y1[ind1,:], y1[[i for i in range(n_points) if i not in ind1],:]
X2_train, X2 = cl2[ind2,:], cl2[[i for i in range(n_points) if i not in ind2],:]
y2_train, y2 = y2[ind2,:], y2[[i for i in range(n_points) if i not in ind2],:]

X_test = np.vstack((X1,X2))
y_test = np.vstack((y1,y2))

X_train = np.vstack((X1_train,X2_train))
y_train = np.vstack((y1_train,y2_train))

X_test_r = np.copy(X_test)
y_test_r = np.copy(y_test)

X_train_r = np.copy(X_train)
y_train_r = np.copy(y_train)

logreg = LogisticRegression()

def clf(X_train, X_test, y_train, y_test):
    
    logreg.fit(X_train, y_train.ravel())

    pred = logreg.predict_proba(X_test)
    sc_c = logreg.score(X_test, y_test)

    w = logreg.coef_[0]
    a = -w[0] / w[1]
    yy = a * xx - (logreg.intercept_[0]) / w[1]

    return pred, sc_c, yy

def xy_update(X_train, X_test, y_train, y_test, ind):

    X_train = np.vstack((X_train, X_test[ind,:]))
    y_train = np.vstack((y_train, y_test[ind,:]))

    X_test = X_test[[i for i in range(len(X_test)) if i != ind],:]
    y_test = y_test[[i for i in range(len(y_test)) if i != ind],:]

    return X_train, X_test, y_train, y_test

n_iter = 50 # number of iterations

sc = []
sc_r = []

xx = np.linspace(-4,4)

for i in range(n_iter):

    # --------------- uncertainty sampling ----------------

    pred, sc_c, yy = clf(X_train, X_test, y_train, y_test)
    
    print('score  : %.6f' % (sc_c))
    sc.append(sc_c)
    
    ind_min_conf = np.argmin(np.absolute(pred-0.5), axis=0)[0]
    # print('Least confident: ', X_test[ind_min_conf,:], y_test[ind_min_conf],'\n')

    
    # ------------------ random sampling ------------------

    pred, sc_c, yy_r = clf(X_train_r, X_test_r, y_train_r, y_test_r)
    
    print('score_r: %.6f\n' % (sc_c))
    sc_r.append(sc_c)

    ind_r = np.random.choice(range(len(X_test_r)))

    # -----------------------------------------------------

    class_colors = ["red" if i else "green" for i in y_train]

    plt.scatter(cl1[:,0], cl1[:,1], s = 2, marker='.', color='green')
    plt.scatter(cl2[:,0], cl2[:,1], s = 2, marker='.', color='red')
    plt.scatter(X_train[:,0], X_train[:,1], s = 50, marker="s", c=class_colors)
    plt.plot(xx, yy, 'b-')
    plt.plot(xx, yy_r, 'b:')
    plt.xlim((-5, 5))   
    plt.ylim((-5, 5))
    plt.title('Iteration '+str(i+1))
    plt.grid()
    plt.show()

    X_train, X_test, y_train, y_test = xy_update(X_train, X_test, y_train, y_test, ind_min_conf)
    X_train_r, X_test_r, y_train_r, y_test_r = xy_update(X_train_r, X_test_r, y_train_r, y_test_r, ind_r)


plt.plot(range(n_iter), sc, 'b-')
plt.plot(range(n_iter), sc_r, 'b:')
plt.grid()
plt.show()