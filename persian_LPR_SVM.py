import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

start = time.process_time()
res = 16
test_frac = 0.4


def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32") / scale


def process_data():
    path_to_data = "./persian_LPR/"
    folder_list = os.listdir(path_to_data)
    sz = (res, res)
    data = []
    data_label = []
    count = 0
    for folder in folder_list:
        path = os.path.join(path_to_data + folder + "/")
        img_list = os.listdir(path)
        for name in img_list:
            img = cv2.imread(path + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = resize_and_scale(img, sz, 255)
            data.append(img.flatten())
            label = folder
            if folder == "W":
                label = "ص"
            if folder == "S":
                label = "س"
            data_label.append(label)
            count += 1

    data = np.array(data)
    data_label = np.array(data_label)

    return data, data_label, count


def unflatten(flattened):
    img = [[0 for x in range(res)] for y in range(res)]
    count = 0
    for i in range(res):
        for j in range(res):
            img[i][j] = 255 - flattened[count] * 255
            count += 1
    img = np.array(img)
    return img


def main():
    data, data_labels, data_size = process_data()

    x_train, x_test, x_train_labels, x_test_correct_labels = train_test_split(data, data_labels, test_size=test_frac,
                                                                              random_state=42, shuffle=True)

    print("Data size:", data_size)
    test_size = int(test_frac * data_size)

    print("Data fetched successfully!")
    clf = svm.SVC(C=1000, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=False, probability=False,
                  tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                  decision_function_shape='ovr', random_state=None)

    clf.fit(x_train, x_train_labels)

    predictions = clf.predict(x_test)

    print("Training set score:", clf.score(x_train, x_train_labels))
    count = 0
    for i in range(test_size):
        if predictions[i] == x_test_correct_labels[i]:
            count += 1
        else:
            imgplot = plt.imshow(unflatten(x_test[i]), cmap="Greys")
            plt.title(("Prediction:", predictions[i], " Correct Label: ", x_test_correct_labels[i]))
            plt.show()
    print("Test accuracy: ", count / test_size)
    print("Wrong predictions: ", test_size - count, "out of", test_size)
    print("Correct predictions: ", count, "out of", test_size)


main()
print("Time elapsed: ", time.process_time() - start, "seconds")

'''cv = GridSearchCV(estimator=svm.SVC(),
                      param_grid={'C': [10, 100, 1000], 'kernel': ('linear', 'rbf', 'sigmoid', 'poly',),
                                  'decision_function_shape': ('ovr', 'ovo'), 'gamma': ('scale', 'auto'),
                                  'shrinking': (False, True), 'tol': [0.001, 0.0001], 'probability': [False, True]})
    cv.fit(x_train, x_train_labels)
    clf = cv.best_estimator_
    print("Parameters of best estimator:", cv.best_params_)'''
