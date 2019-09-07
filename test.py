import numpy as np
import cv2
import keras
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import img_to_array
import pickle

def test():
    haar_file = 'haarcascade_frontalface_default.xml'
    model = load_model('models/model.h5')
    file = open("models/list.pkl","rb")
    customer_list = pickle.load(file)
    print(model.summary())
    print(customer_list)
    face_cascade = cv2.CascadeClassifier(haar_file)
    target_size = 64
    webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this
    # The program loops until it has 30 images of the face.
    count = 1
    while count < 20: 
        (_, im) = webcam.read()
        faces = face_cascade.detectMultiScale(im, 1.3, 4)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = im[y:y + h, x:x + w]
            print(np.shape(face))
            image = img_to_array(face)
            image = cv2.resize(image,(target_size,target_size))
            image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
            # print(np.shape(image))
            result = model.predict_classes(image)
            print('Predict',result)
            print('user list',customer_list)
            
            name = customer_list[result[0]]

            # name = test(face,model,target_size,auth_list)
            #         info = retrive_data()
            print('Name',name)
            if name == 'unknown':
                count = 1
            else : count +=1
        
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    webcam.release()
    cv2.destroyAllWindows()
    return name

# def test(status):
#     if status == 'customer':

#         haar_file = 'haarcascade_frontalface_default.xml'
#         model = load_model('models/cust_model.h5')
#         file = open("models/cust_list.pkl","rb")
#         customer_list = pickle.load(file)
#         print(model.summary())
#         print(customer_list)
#         face_cascade = cv2.CascadeClassifier(haar_file)
#         target_size = 64
#         webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this
#         # The program loops until it has 30 images of the face.
#         count = 1
#         while count < 30: 
#             (_, im) = webcam.read()
#             faces = face_cascade.detectMultiScale(im, 1.3, 4)
#             for (x,y,w,h) in faces:
#                 cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
#                 face = im[y:y + h, x:x + w]
#                 print(np.shape(face))
#                 image = img_to_array(face)
#                 image = cv2.resize(image,(target_size,target_size))
#                 image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
#                 # print(np.shape(image))
#                 result = model.predict_classes(image)
#                 print('Predict',result)
#                 print('user list',customer_list)
                
#                 name = customer_list[result[0]]

#                 # name = test(face,model,target_size,auth_list)
#                 #         info = retrive_data()
#                 print('Name',name)
#                 if name == 'unknown':
#                     count = 1
#                 else : count +=1
            
#             cv2.imshow('OpenCV', im)
#             key = cv2.waitKey(10)
#             if key == 27:
#                 break
#         webcam.release()
#         cv2.destroyAllWindows()
               
#     else:
#         haar_file = 'haarcascade_frontalface_default.xml'
#         model = load_model('models/auth_model.h5')
#         file = open("models/auth_list.pkl","rb")
#         customer_list = pickle.load(file)
#         print(model.summary())
#         print(customer_list)
#         face_cascade = cv2.CascadeClassifier(haar_file)
#         target_size = 64
#         webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this
#         # The program loops until it has 30 images of the face.
#         count = 1
#         while count < 30: 
#             (_, im) = webcam.read()
#             faces = face_cascade.detectMultiScale(im, 1.3, 4)
#             for (x,y,w,h) in faces:
#                 cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
#                 face = im[y:y + h, x:x + w]
#                 print(np.shape(face))
#                 image = img_to_array(face)
#                 image = cv2.resize(image,(target_size,target_size))
#                 image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
#                 # print(np.shape(image))
#                 result = model.predict_classes(image)
#                 print('Predict',result)
#                 print('user list',customer_list)
                
#                 name = customer_list[result[0]]

#                 # name = test(face,model,target_size,auth_list)
#                 #         info = retrive_data()
#                 print('Name',name)
#                 if name == 'unknown':
#                     count = 1
#                 else : count +=1
        
#             cv2.imshow('OpenCV', im)
#             key = cv2.waitKey(10)
#             if key == 27:
#                 break
#         webcam.release()
#         cv2.destroyAllWindows()
               
#     return name


# if __name__ == '__main__':
#     file = open("models/auth_list.pkl","rb")
#     list = pickle.load(file)
#     print(list)
#     print(list[0])
#     print(list[0])