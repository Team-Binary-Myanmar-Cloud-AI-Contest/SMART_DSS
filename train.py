import cv2
import sys
import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import pickle

def model_train():

    datasets = 'datasets'

    (images, labels, names, label) = ([], [], [], 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            if subdir == '.DS_Store':
                continue
            names.append(subdir)
            subjectpath = os.path.join(datasets, subdir)
            print('subjectpath',subjectpath)
            for filename in os.listdir(subjectpath):
                if filename == '.DS_Store':
                    continue
                path = subjectpath + '/' + filename
                print('Path',path)
                imgRead = load_img(path,target_size = (64,64))
                imgRead = img_to_array(imgRead)
                images.append(imgRead)
                labels.append(int(label))
                print(label)
            label += 1
    print(labels)
    print(np.shape(images))
    print(np.shape(labels))
    print(names)
    (width, height) = (130, 100)

    # Create a Numpy array from the two lists above
    (images, labels) = [np.array(lis) for lis in [images, labels]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    X_train = np.array(images)
    Y_train = np.array(labels) 

    nb_classes = label
    Y_train = to_categorical(Y_train, nb_classes)
    # Y_test = to_categorical(Y_test, nb_classes)
    input_shape = (64, 64, 3)

    X_train = X_train.astype('float32')
    X_train /= 255

    # BUILD THE MODEL
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # TRAIN THE MODEL
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(model.summary())
    model.fit(X_train, Y_train, batch_size=5, epochs=30, verbose=1)

    model.save('models/model.h5')
    print("saved the model ______________")
    pickle.dump(names, open("models/list.pkl", "wb"))
    print("saved the PKL PKL  ______________")

# def customer_train():

#     datasets = 'datasets'

#     (images, labels, names, label) = ([], [], [], 0)
#     for (subdirs, dirs, files) in os.walk(datasets):
#         for subdir in dirs:
#             if subdir == '.DS_Store':
#                 continue
#             names.append(subdir)
#             subjectpath = os.path.join(datasets, subdir)
#             print('subjectpath',subjectpath)
#             for filename in os.listdir(subjectpath):
#                 if filename == '.DS_Store':
#                     continue
#                 path = subjectpath + '/' + filename
#                 print('Path',path)
#                 imgRead = load_img(path,target_size = (64,64))
#                 imgRead = img_to_array(imgRead)
#                 images.append(imgRead)
#                 labels.append(int(label))
#                 print(label)
#             label += 1
#     print(labels)
#     print(np.shape(images))
#     print(np.shape(labels))
#     print(names)
#     (width, height) = (130, 100)

#     # Create a Numpy array from the two lists above
#     (images, labels) = [np.array(lis) for lis in [images, labels]]

#     # OpenCV trains a model from the images
#     # NOTE FOR OpenCV2: remove '.face'
#     X_train = np.array(images)
#     Y_train = np.array(labels) 

#     nb_classes = label
#     Y_train = to_categorical(Y_train, nb_classes)
#     # Y_test = to_categorical(Y_test, nb_classes)
#     input_shape = (64, 64, 3)

#     X_train = X_train.astype('float32')
#     X_train /= 255

#     # BUILD THE MODEL
#     model = Sequential()

#     model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(32, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Convolution2D(64, 3, 3, border_mode='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(64, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Convolution2D(64, 3, 3, border_mode='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(64, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(nb_classes))
#     model.add(Activation('softmax'))

#     # TRAIN THE MODEL
#     adam = Adam(lr=0.0001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#     print(model.summary())
#     model.fit(X_train, Y_train, batch_size=5, epochs=30, verbose=1)

#     model.save('models/cust_model.h5')
#     print("saved the model ______________")
#     pickle.dump(names, open("models/cust_list.pkl", "wb"))
#     print("saved the PKL PKL  ______________")

# def auth_train():

#     datasets = 'auth-data'

#     (images, labels, names, label) = ([], [], [], 0)
#     for (subdirs, dirs, files) in os.walk(datasets):
#         for subdir in dirs:
#             if subdir == '.DS_Store':
#                 continue
#             names.append(subdir)
#             subjectpath = os.path.join(datasets, subdir)
#             print('subjectpath',subjectpath)
#             for filename in os.listdir(subjectpath):
#                 if filename == '.DS_Store':
#                     continue
#                 path = subjectpath + '/' + filename
#                 print('Path',path)
#                 imgRead = load_img(path,target_size = (64,64))
#                 imgRead = img_to_array(imgRead)
#                 images.append(imgRead)
#                 labels.append(int(label))
#                 print(label)
#             label += 1
#     print(labels)
#     print(np.shape(images))
#     print(np.shape(labels))
#     print(names)
#     (width, height) = (130, 100)

#     # Create a Numpy array from the two lists above
#     (images, labels) = [np.array(lis) for lis in [images, labels]]

#     # OpenCV trains a model from the images
#     # NOTE FOR OpenCV2: remove '.face'
#     X_train = np.array(images)
#     Y_train = np.array(labels) 

#     nb_classes = label
#     Y_train = to_categorical(Y_train, nb_classes)
#     # Y_test = to_categorical(Y_test, nb_classes)
#     input_shape = (64, 64, 3)

#     X_train = X_train.astype('float32')
#     X_train /= 255

#     # BUILD THE MODEL
#     model = Sequential()

#     model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(32, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Convolution2D(64, 3, 3, border_mode='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(64, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Convolution2D(64, 3, 3, border_mode='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(64, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(nb_classes))
#     model.add(Activation('softmax'))

#     # TRAIN THE MODEL
#     adam = Adam(lr=0.0001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#     print(model.summary())
#     model.fit(X_train, Y_train, batch_size=5, epochs=30, verbose=1)

#     model.save('models/auth_model.h5')
#     pickle.dump(names, open("models/auth_list.pkl", "wb"))

if __name__ == '__main__':
    customer_train()