import numpy as np
import os
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt

start = time.process_time()
res = 16
test_size = 2007
path = "./USPS_images"

def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32") / scale


def process_train_data():
    '''TRAIN DATA'''
    path_to_data = os.path.join(path, "train/")
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
            label = name.split("_")
            validation_usps.append(resized_img.flatten())
            validation_usps_label.append(label[0])
            validation_usps_count.append(label[1])

    validation_usps = np.array(validation_usps)
    validation_usps_label = np.array(validation_usps_label)
    validation_usps_count = np.array(validation_usps_count)
    return validation_usps, validation_usps_label, validation_usps_count


def process_test_data():
    '''TEST DATA'''
    path_to_data =os.path.join(path, "test/")
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
    x_train, x_train_labels, x_train_count = process_train_data()
    x_test, x_test_correct_labels, x_test_count, x_bitmap = process_test_data()

    print("Data fetched succesfully!")

    mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', alpha=1e-4,
                        solver='sgd', tol=1e-4, random_state=1, max_iter=500,
                        learning_rate_init=.1, verbose=True, early_stopping=False)

    mlp.fit(x_train, x_train_labels)
    print("Training set score: %f" % mlp.score(x_train, x_train_labels))

    predictions = mlp.predict(x_test)
    count = 0
    percentages = [[0 for i in range(10)] for j in range(2)]
    for i in range(test_size):
        percentages[0][int(x_test_correct_labels[i])] += 1
        if predictions[i] == x_test_correct_labels[i]:
            count += 1
            percentages[1][int(x_test_correct_labels[i])] += 1
        else:
            imgplot = plt.imshow(x_bitmap[i], cmap="Greys")
            plt.title(("Prediction:", predictions[i], " Correct Label: ", x_test_correct_labels[i],
                       " Pic number:", x_test_count[i]))
            plt.show()

    print("Test accuracy: ", count / test_size)
    print("Wrong predictions: ", test_size - count, "out of", test_size)
    print("Correct predictions: ", count, "out of", test_size)
    for i in range(10):
        print("Digit:", i, ": total:", percentages[0][i], "correct:", percentages[1][i], "  accuracy:",
              percentages[1][i] / percentages[0][i])


main()
print("Time elapsed: ", time.process_time() - start, "seconds")

'''   parameters = {'hidden_layer_sizes': [25, 50], 'activation': ('logistic', 'relu', 'tanh'),
                  'alpha': [0.01, 0.0001], 'solver': ('sgd', 'adam', 'lbfgs'), 'early_stopping': (False, True),
                  'warm_start': (True, False), 'max_iter':[500, 5000]}

    cv = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters)
    cv.fit(x_train, x_train_labels)
    mlp = cv.best_estimator_
    print("Parameters of best estimator:", cv.best_params_)'''
