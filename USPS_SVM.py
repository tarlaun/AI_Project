import numpy as np
import os
import cv2
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt

start = time.process_time()
res = 16
test_size = 2007


def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32") / scale


def process_usps_data():
    '''TRAIN DATA'''
    path_to_data = "./USPS_images/train/"
    img_list = os.listdir(path_to_data)
    sz = (res, res)
    validation_usps = []
    validation_usps_label = []
    validation_usps_count = []
    for name in img_list:
        if '.jpg' in name:
            img = cv2.imread(path_to_data + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = resize_and_scale(img, sz, 255)
            validation_usps.append(resized_img.flatten())
            label = name.split("_")
            validation_usps_label.append(label[0])
            validation_usps_count.append(label[1])

    validation_usps = np.array(validation_usps)
    validation_usps_label = np.array(validation_usps_label)
    validation_usps_count = np.array(validation_usps_count)
    return validation_usps, validation_usps_label, validation_usps_count


def process_test_data():
    '''TEST DATA'''
    path_to_data = "./USPS_images/test/"
    img_list = os.listdir(path_to_data)
    sz = (res, res)
    validation_usps = []
    validation_usps_label = []
    validation_usps_count = []
    x_bitmap = []
    for name in img_list:
        if '.jpg' in name:
            img = cv2.imread(path_to_data + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = resize_and_scale(img, sz, 255)
            validation_usps.append(resized_img.flatten())
            x_bitmap.append(resized_img)
            label = name.split("_")
            validation_usps_label.append(label[0])
            validation_usps_count.append(label[1])
    validation_usps = np.array(validation_usps)
    validation_usps_label = np.array(validation_usps_label)
    validation_usps_count = np.array(validation_usps_count)
    return validation_usps, validation_usps_label, validation_usps_count, x_bitmap


def main():
    x_train, x_train_labels, x_train_count = process_usps_data()
    x_test, x_test_correct_labels, x_test_count, x_bitmap = process_test_data()

    print("Data fetched succesfully!")

    clf = svm.SVC(C=10, kernel='rbf', decision_function_shape='ovr', gamma='scale', shrinking=False)
    clf.fit(x_train, x_train_labels)

    predictions = clf.predict(x_test)

    print("Training set score:", clf.score(x_train, x_train_labels))
    count = 0
    for i in range(test_size):
        if predictions[i] == x_test_correct_labels[i]:
            count += 1
        else:
            imgplot = plt.imshow(x_bitmap[i], cmap="Greys")
            plt.title(("Prediction:", predictions[i], " Correct Label: ", x_test_correct_labels[i],
            " Pic number: ", x_test_count[i]))
            plt.show()
    print("Test accuracy: ", count / test_size)
    print("Wrong predictions: ", test_size - count, "out of ", test_size)
    print("Correct predictions: ", count, "out of ", test_size)


main()
print("Time elapsed: ", time.process_time() - start, "seconds")

''' clf = svm.SVC(C=10, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=False, probability=False,
               tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
               decision_function_shape='ovr', random_state=None)


 cv = GridSearchCV(estimator=svm.SVC(),
                   param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf', 'sigmoid', 'poly',),
                               'decision_function_shape': ('ovr', 'ovo'), 'gamma': ('scale', 'auto'),
                               'shrinking': (False, True), })
 cv.fit(x_train, x_train_labels)
 clf = cv.best_estimator_
 print("Parameters of best estimator:", cv.best_params_)'''