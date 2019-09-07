from flask import Flask, render_template, redirect, url_for
from flask import request
import os
import cv2
import pickle
from create_data import create_data
from test import test
from retrieve import retrieve
from keras.models import load_model
from train import model_train
import random

app = Flask(__name__)

@app.route('/front_face/<directory>')
def frontal_face(directory):
    return render_template('frontal_face.html',directory=directory)

@app.route('/left_face/<directory>')
def left_face(directory):
    return render_template('left_face.html',directory=directory)

@app.route('/right_face/<directory>')
def right_face(directory):
    return render_template('right_face.html',directory=directory)

@app.route('/up_face/<directory>')
def up_face(directory):
    return render_template('up_face.html',directory=directory)

@app.route('/down_face/<directory>')
def down_face(directory):
    return render_template('down_face.html',directory=directory)

# @app.route('/<directory>', methods=['GET'])
# def index(directory):
#     print('Directory in index:', directory)
#     return render_template('index.html',name=directory)

@app.route('/train')
def train(name=None):
    model_train()
    print("Done Training ____________________")
    return render_template('home.html',flag=True)

@app.route('/create_frontal_data/<directory>')
def create_frontal_data(directory):
    flag='front'
    # import create_data
    create_data(directory,flag)
    print('folder name',directory)
    # folder = name
    # print('folder',folder)
    print("done")
    return redirect(url_for('left_face',directory = directory))

@app.route('/create_left_data/<directory>')
def create_left_data(directory):
    flag='left'
    # import create_data
    create_data(directory,flag)
    print('folder name',directory)
    # folder = name
    # print('folder',folder)
    print("done")
    return redirect(url_for('right_face',directory = directory))

@app.route('/create_right_data/<directory>')
def create_right_data(directory):
    flag='right'
    # import create_data
    create_data(directory,flag)
    print('folder name',directory)
    # folder = name
    # print('folder',folder)
    print("done")
    return redirect(url_for('up_face',directory = directory))

@app.route('/create_up_data/<directory>')
def create_up_data(directory):
    flag='up'
    # import create_data
    create_data(directory,flag)
    print('folder name',directory)
    # folder = name
    # print('folder',folder)
    print("done")
    return redirect(url_for('down_face',directory = directory))

@app.route('/create_down_data/<directory>')
def create_down_data(directory):
    flag='down'
    # import create_data
    create_data(directory,flag)
    print('folder name',directory)
    # folder = name
    # print('folder',folder)
    auth_name = 'again'
    print("done")
    return redirect(url_for('home',auth_name=auth_name))

@app.route('/register')
def register():
    return render_template('register.html')

@app.route("/post_field", methods=["POST"])
def need_input(name=None):
    # form = request.form()
    info = []
    random_number = random.randrange(10000,99999)
    customer_id = 'ID'+str(random_number)
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    phone_number = request.form['phone_number']
    # points = request.form['points']
    name = firstname+' '+lastname
    print(customer_id)
    print(firstname)
    print(lastname)
    print(phone_number)
    # print(points)
    info.append(customer_id)
    info.append(firstname)
    info.append(lastname)
    info.append(phone_number)
    # info.append(points)
    customer_info_name = 'info/'+name+'.pkl'

    pickle.dump(info, open(customer_info_name, "wb"))
    print("saved customer info info  ______________")
    # for key, value in request.form.items():
    #     if key=='firstname':
    #         directory = value
    #         print('Directory in post: ',directory)
    return redirect(url_for('frontal_face',directory = name))
    # return render_template('login.html')

@app.route('/')
def initial():
    return render_template('login.html')

@app.route('/home/<auth_name>',methods=['GET'])
def home(auth_name):
    return render_template('home.html',auth_name=auth_name)

@app.route('/loginfail')
def loginfail():
    return render_template('fail.html')

# @app.route('/login')
# def login():
#     haar_file = 'haarcascade_frontalface_default.xml'
#     face_cascade = cv2.CascadeClassifier(haar_file)
#     model = load_model('models/auth_model.h5')
#     file = open("models/auth_list.pkl","rb")
#     auth_list = pickle.load(file)
#     print(model.summary())
#     print(auth_list)
#     target_size = 64
#     webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this
#     # The program loops until it has 30 images of the face.
#     count = 1
#     while count < 30: 
#         (_, im) = webcam.read()
#         faces = face_cascade.detectMultiScale(im, 1.3, 4)
#         for (x,y,w,h) in faces:
#             cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
#             face = im[y:y + h, x:x + w]
#             name = test(face,model,target_size,auth_list)
#             print('Name',name)
#             if name == 'unknown':
#                 count = 1
#             else : count +=1
        
#         cv2.imshow('OpenCV', im)
#         key = cv2.waitKey(10)
#         if key == 27:
#             break
#     webcam.release()
#     cv2.destroyAllWindows()
#     return redirect(url_for('home',auth_name=name))

# @app.route('/payment')
# def payment():
#     haar_file = 'haarcascade_frontalface_default.xml'
#     model = load_model('models/auth_model.h5')
#     file = open("models/auth_list.pkl","rb")
#     customer_list = pickle.load(file)
#     print(model.summary())
#     print(customer_list)
#     face_cascade = cv2.CascadeClassifier(haar_file)
#     target_size = 64
#     webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this
#     # The program loops until it has 30 images of the face.
#     count = 1
#     while count < 30: 
#         (_, im) = webcam.read()
#         faces = face_cascade.detectMultiScale(im, 1.3, 4)
#         for (x,y,w,h) in faces:
#             cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
#             face = im[y:y + h, x:x + w]
#             name = test(face,model,target_size,auth_list)
#             info = retrive_data()
#             print('Name',name)
#             if name == 'unknown':
#                 count = 1
#             else : count +=1
        
#         cv2.imshow('OpenCV', im)
#         key = cv2.waitKey(10)
#         if key == 27:
#             break
#     webcam.release()
#     cv2.destroyAllWindows()
#     return redirect(url_for('info',info_list=info))

@app.route('/login')
def login():
    name = test()
    print('Name',name)
    if name == 'phoo pyae pyae linn':
        return redirect(url_for('home',auth_name=name))
    else :
        return redirect(url_for('loginfail'))

@app.route('/cashier')
def cashier():
    name = test()
    print('Name',name)
    # status = 'customer'
    # test.test(status)
    # info = retrieve()
    # return redirect(url_for('home',auth_name='again'))
    return redirect(url_for('info',customer_name=name))


@app.route('/info/<customer_name>',methods=['GET'])
def info(customer_name):
    info = retrieve(customer_name)
    print('info',info)
    print('first name',info[0])
    print('last name',info[1])
    print('phone number',info[2])
    # print('points',info[3])
    return render_template('info.html',info_list=info)

@app.route("/confirmation", methods=["POST"])
def confirmation(name=None):
    # form = request.form()
    info_list =[]
    customer_id = request.form['customer_id']
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    phone_number = request.form['phone_number']
    invoice_id = request.form['invoice_id']
    # points = request.form['points']
    name = firstname+' '+lastname
    frequent_come = 1
    print(customer_id)
    print(firstname)
    print(lastname)
    print(phone_number)
    print(invoice_id)
    # print(points)
    # points = int(points) - int(amount)

    info_list.append(customer_id)
    info_list.append(firstname)
    info_list.append(lastname)
    info_list.append(phone_number)
    info_list.append(invoice_id)
    customer_info_name = 'info/'+name+'.pkl'

    pickle.dump(info_list, open(customer_info_name, "wb"))
    print("saved customer info info  ______________")

    return redirect(url_for('home',auth_name='again'))
    # for key, value in request.form.items():
    #     if key=='firstname':
    #         directory = value
    #         print('Directory in post: ',directory)
    #return redirect(url_for('confirmation',name=name,firstname=firstname,lastname=lastname,phone_number=phone_number,points=points))
    # return render_template('login.html')

# @app.route("/pay", methods=["POST"])
# def pay(name=None):
#     print('confirmation')
#     sample = request.form['sample']
#     print(sample)
#     return redirect(url_for('home',auth_name='again'))

# @app.route('/confirmation/<name>/<firstname>/<lastname>/<phone_number>/<points>')
# def confirmation(name,firstname,lastname,phone_number,points):
#     # info = retrieve(name)
#     print('Name',name)
#     print('firstname',firstname)
#     print('last name',lastname)
#     print('phone number',phone_number)
#     print('points',points)

#     info_list.append(firstname)
#     info_list.append(lastname)
#     info_list.append(phone_number)
#     info_list.append(points)

#     customer_info_name = 'info/'+name+'.pkl'

#     pickle.dump(info_list, open(customer_info_name, "wb"))
#     print("saved customer info info  ______________")
#     # status = 'customer'
#     # test.test(status)
#     # info = retrieve()
#     # return redirect(url_for('home',auth_name='again'))
#     return redirect(url_for('home',auth_name='again'))

if __name__ == '__main__':
    app.run()
else :
    from werkzeug.debug import DebuggedApplication
    app.wsgi_app = DebuggedApplication(app.wsgi_app,True)
    app.debug = True