import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import time

np.random.seed(3)
train_size = 1000
test_size = 1000
x_max = 5
x_min = -5
y_max = 5
y_min = -5
start = time.process_time()


def func(x, y):
    z = 4 * np.sin(2 * x)
    if y > z:
        return 1
    return -1


x = [[0 for i in range(2)] for j in range(train_size)]

x0 = np.random.uniform(x_min, x_max, size=train_size)
x1 = np.random.uniform(y_min, y_max, size=train_size)

for i in range(train_size):
    x[i][0] = x0[i]
    x[i][1] = x1[i]
label = [0 for i in range(train_size)]
for i in range(train_size):
    label[i] = func(x[i][0], x[i][1])
print(x)
x_test = [[0 for i in range(2)] for j in range(test_size)]
x0 = np.random.uniform(x_min, x_max, size=test_size)
x1 = np.random.uniform(y_min, y_max, size=test_size)
x_test_labels = [0 for i in range(test_size)]
for i in range(test_size):
    x_test[i][0] = x0[i]
    x_test[i][1] = x1[i]
    x_test_labels[i] = func(x0[i], x1[i])

clf = svm.SVC(C=1000, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=False, probability=False,
              tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
              decision_function_shape='ovr', random_state=None)
clf.fit(x, label)
pred = clf.predict(x_test)

x = np.array(x)
x_test = np.array(x_test)

plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=label, zorder=10, cmap=plt.cm.Paired,
            edgecolor='k', s=20)

plt.scatter(x_test[:, 0], x_test[:, 1], s=80, facecolors='none',
            zorder=10, edgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], c=pred, zorder=10, cmap=plt.cm.Paired,
            edgecolor='k', s=20)

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')

# plot the line, the points, and the nearest vectors to the plane
#plotting the margin for linear kernel
'''w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')'''

plt.show()
print("Train accuracy:", clf.score(x, label))
print("Test accuracy:", clf.score(x_test, x_test_labels))
print("Time elapsed:", time.process_time() - start, "seconds")
