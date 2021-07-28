import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Perceptron(object):
    def __init__(self, LR=0.5, i=10):
        self.LR = LR
        self.i = i
        self.errors = []
        self.loadofnode = None
        self.bs = None
        self._af_func = self._af_func

    def fit(self, x, y):
        self.loadofnode = np.zeros(x.shape[1])
        self.bs = 0.0
        for i in range(self.i):
            error = 0
            for xi, target in zip(x, y):
                l_out = self.summat(xi)
                ytarget = self._af_func(l_out)
                change = self.LR * (target - ytarget)
                self.loadofnode += change * xi
                self.bs += change
                error += int(change != 0)
            self.errors.append(error)
        return self

    def guess(self, X):
        l_out = np.dot(X, self.loadofnode) + self.bs
        ytarget = self._af_func(l_out)
        return ytarget
    def summat(self, x):return np.dot(x, self.loadofnode) + self.bs
    def _af_func(self, x):return np.where(x >= 0, 1, 0)

iris = pd.read_csv('iris.csv')
modifiedIris = pd.read_csv('irismodified.csv')
iris.head()
y = iris.iloc[:, 4].values
x = iris.iloc[:, 0:3].values
fig = plt.figure()
d3 = plt.axes(projection='3d')
d3.set_title('Iris dataset')
d3.set_xlabel("Sepal l")
d3.set_ylabel("Sepal w")
d3.set_zlabel("Petal")
d3.scatter(x[:50, 0], x[:50, 1], x[:50, 2], color='black',marker='v', s=4, edgecolor='yellow', label="Iris Setosa")
d3.scatter(x[50:100, 0], x[50:100, 1], x[50:100, 2], color='grey',marker='^', s=4, edgecolor='purple', label="Iris Versicolour")
plt.legend(loc='upper left')
plt.show()

x = x[0:100, 0:2]
y = y[0:100]
plt.scatter(x[:50, 0], x[:50, 1], color='cyan', marker='s', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='magenta', marker='v',label='Versicolour')
plt.xlabel("Sepal for iris dataset")
plt.ylabel("Petal for iris dataset")
plt.legend(loc='upper left')
plt.show()

y2 = modifiedIris.iloc[:, 4].values
x2 = modifiedIris.iloc[:, 0:3].values
x2 = x2[0:100, 0:2]
y2 = y2[0:100]
plt.scatter(x2[:50, 0], x2[:50, 1], color='cyan', marker='s', label='Setosa')
plt.scatter(x2[50:100, 0], x2[50:100, 1], color='magenta', marker='v',label='Versicolour')
plt.xlabel("Sepal for modified iris dataset")
plt.ylabel("Petal for modified iris dataset")
plt.legend(loc='upper left')
plt.show()

from sklearn.model_selection import train_test_split

y = np.where(y == 'Iris-setosa', 1, 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)
classi = Perceptron(LR=0.001, i=26)
classi.fit(x_train, y_train)
plt.plot(range(1, len(classi.errors) + 1),classi.errors, marker='^',color='green')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(x, y, classi, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classi.guess(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    plt.show()

plot_decision_regions(x_test, y_test, classi)
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
print("accuracy %f" % accuracy_score(classi.guess(x_test), y_test))
print("R2 score:", r2_score(y_test, classi.guess(x_test)))