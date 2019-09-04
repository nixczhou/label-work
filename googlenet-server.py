import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

import sys
import numpy as np


from PIL import Image
import glob

import cv2
import numpy as np

import json
import math

x = 0
y = 0

status = []
windows = []
image_list1 = []
image_list2 = []
image_list3 = []
count = 0
uploaded_image = 0
save_list = []

# Global Constants
NB_CLASS=2
LEARNING_RATE=0.0001
MOMENTUM=0.9
ALPHA=0.0001
BETA=0.75
GAMMA=0.1
DROPOUT=0.4
WEIGHT_DECAY=0.0005
LRN2D_NORM=True
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
USE_BN=True
IM_WIDTH=224
IM_HEIGHT=224
EPOCH=100

train_root='static/infer/train/'
vaildation_root='static/infer/valid/'
test_root='static/infer/test/'

IM_WIDTH=224
IM_HEIGHT=224
batch_size=32

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

#normalization
def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY):
    #l2 normalization
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    if lrn2d_norm:
        #batch normalization
        x=BatchNormalization()(x)

    return x



def inception_module(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=None):
    (branch1,branch2,branch3,branch4)=params
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None
    #1x1
    pathway1=Conv2D(filters=branch1[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    #1x1->3x3
    pathway2=Conv2D(filters=branch2[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway2=Conv2D(filters=branch2[1],kernel_size=(3,3),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway2)

    #1x1->5x5
    pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)

    #3x3->1x1
    pathway4=MaxPooling2D(pool_size=(3,3),strides=1,padding=padding,data_format=DATA_FORMAT)(x)
    pathway4=Conv2D(filters=branch4[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway4)

    return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)



def create_model():
    #Data format:tensorflow,channels_last;theano,channels_last
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,224,224)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(224,224,3)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3
    else:
        raise Exception('Invalid Dim Ordering')

    x=conv2D_lrn2d(img_input,64,(7,7),2,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    x=BatchNormalization()(x)

    x=conv2D_lrn2d(x,64,(1,1),1,padding='same',lrn2d_norm=False)

    x=conv2D_lrn2d(x,192,(3,3),1,padding='same',lrn2d_norm=True)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(64,),(96,128),(16,32),(32,)],concat_axis=CONCAT_AXIS) #3a
    x=inception_module(x,params=[(128,),(128,192),(32,96),(64,)],concat_axis=CONCAT_AXIS) #3b
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(192,),(96,208),(16,48),(64,)],concat_axis=CONCAT_AXIS) #4a
    x=inception_module(x,params=[(160,),(112,224),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4b
    x=inception_module(x,params=[(128,),(128,256),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4c
    x=inception_module(x,params=[(112,),(144,288),(32,64),(64,)],concat_axis=CONCAT_AXIS) #4d
    x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #4e
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #5a
    x=inception_module(x,params=[(384,),(192,384),(48,128),(128,)],concat_axis=CONCAT_AXIS) #5b
    x=AveragePooling2D(pool_size=(7,7),strides=1,padding='valid',data_format=DATA_FORMAT)(x)

    x=Flatten()(x)
    x=Dropout(DROPOUT)(x)
    x=Dense(output_dim=NB_CLASS,activation='linear')(x)
    x=Dense(output_dim=NB_CLASS,activation='softmax')(x)

    return x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT


def check_print():
    # Create the Model
    x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT=create_model()

    # Create a Keras Model
    model=Model(input=img_input,output=[x])
    model.summary()

    # Save a PNG of the Model Build
    plot_model(model,to_file='GoogLeNet.png')

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc',metric.top_k_categorical_accuracy])
    print ('Model Compiled')
    return model


class edit:
    def __init__(self, filename, w, h):
        self.filename = filename
        self.w = w
        self.h = h

    def crop(self, x, y):
        img = cv2.imread(self.filename)
        height, width= img.shape[:2]
        names = self.filename.split("/")

        #print(height, width)

        # Set the center of the cropped image coordinate
        x_input = x
        y_input = y

        # 裁切區域的 x 與 y 座標（左上角）
        x = int(width/2 + x_input)
        y = int(height/2 + y_input)

        new_w = int(self.w/2)
        new_h = int(self.h/2)

        crop_img1 = img[y-new_h:y+new_h, x-new_w:x+new_w]

        return crop_img1

    def rotate(self, image, angle):
        center = (self.w/2, self.h/2)
        scale = 1
        left = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_left = cv2.warpAffine(image, left, (self.w, self.h))
        return rotated_left



# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    global windows, status, x, y, count, uploaded_image
    status = []
    windows = []
    image_list1 = []
    image_list2 = []
    image_list3 = []
    count = 0
    x = 0
    y = 0
    uploaded_image = 0
    with open('config.json') as config_file:
        data = json.load(config_file)
    for i in range(len(data['window'])):
        windows.append(data['window'][i])
    print('Setting Files :')
    print(windows)
    for filename in glob.glob('uploads/*.png'):
        os.remove(filename)
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded files
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Move the file form the temporal folder to the upload
            # folder we setup
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Save the filename into a list, we'll use it later
            filenames.append(filename)
            # Redirect the user to the uploaded_file route, which
            # will basicaly show on the browser the uploaded file
    # Load an html page with a link to each uploaded file
    return render_template('upload.html', filenames=filenames)

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/crop')
def crop():
    global x, y, status, uploaded_image, save_list
    x_json = []
    y_json = []
    w_json = []
    h_json = []
    image_list1 = []
    image_list3 = []
    windowss = []

    print('count: ' + str(count))

    print('Click_Position_x : ', x)
    print('Click_Position_y : ', y)
    if (x == 0 and y == 0): #When first launch the crop page
        status = []
        save_list = []
        with open('config.json') as config_file:
            data = json.load(config_file)
        for i in range(len(data['window'])):
            windowss.append(data['window'][i])

        for filename in glob.glob('uploads/*.png'): #assuming jpg(can be png, gif, etc..)
            image_list3.append(filename)

        uploaded_image = len(image_list3)

        #print("Original Image List:")
        #print(image_list3)
        
        for i in range(len(windowss)):
            x_json.append(windowss[i]['posx'])
            y_json.append(windowss[i]['posy'])
            w_json.append(windowss[i]['width'])
            h_json.append(windowss[i]['height'])
            crop_img = edit(image_list3[count],windowss[i]['width'],windowss[i]['height']).crop(windowss[i]['posx'],windowss[i]['posy'])
            rotate_img = edit(image_list3[count],windowss[i]['width'],windowss[i]['height']).rotate(crop_img, windowss[i]['rotation'])
            names = image_list3[count].split("/")
            directory = 'static/datasets/crop/' + windowss[i]['name'] + '/'+ windowss[i]['name'] + '_' + names[1]
            cv2.imwrite(directory, rotate_img)
            #print(directory)
            img_name = names[1]
            status.append('OK')
            image_list1.append(directory)
            save_list.append(directory)

        #print(image_list1)
    else:
        with open('config.json') as config_file:
            data = json.load(config_file)
        for i in range(len(data['window'])):
            windowss.append(data['window'][i])

        for filename in glob.glob('uploads/*.png'): #assuming jpg(can be png, gif, etc..)
            image_list3.append(filename)

        distance = []
        #print("Original Image List:")
        #print(image_list3)
        x_new = float(x) - 250
        y_new = float(y) - 100

        for i in range(len(windowss)):
            x_json.append(windowss[i]['posx'])
            y_json.append(windowss[i]['posy'])
            w_json.append(windowss[i]['width'])
            h_json.append(windowss[i]['height'])
            crop_img = edit(image_list3[count],windowss[i]['width'],windowss[i]['height']).crop(windowss[i]['posx'],windowss[i]['posy'])
            rotate_img = edit(image_list3[count],windowss[i]['width'],windowss[i]['height']).rotate(crop_img, windowss[i]['rotation'])
            names = image_list3[count].split("/")
            directory = 'static/datasets/crop/' + windowss[i]['name'] + '/'+ windowss[i]['name'] + '_' + names[1]
            cv2.imwrite(directory, rotate_img)
            #print(directory)
            img_name = names[1]
            image_list1.append(directory)

        for i in range(len(windowss)):
            distance.append(math.sqrt(pow((windowss[i]['posx'] - x_new),2) + pow((windowss[i]['posy'] -y_new),2)))

        minimum = distance.index(min(distance))
        print("Selected : " + str(minimum))
        
        for i in range(len(distance)):
            if (i == minimum):
                if(status[i] == 'NG'):
                    status[i] = 'OK'
                else:
                    status[i] = 'NG'

    print(status)
    return render_template("crop.html", all_image = image_list1, original= image_list3[count], status = status, count = count, uploaded_image=uploaded_image, x_json = x_json , y_json = y_json, w_json = w_json, h_json = h_json)

@app.route('/calculate', methods=['POST'])
def calculate():
    global x, y
    position = request.form.getlist("coordinate[]")

    print(position)
    x = position[0]
    y = position[1]
    #print(x)
    #print(y)
    return "nothing"

@app.route('/done',  methods=['POST'])
def done():
    new_count = request.form["count"]
    global count, x, y, uploaded_image, save_list
    count = int(new_count)
    print('************SAVE LIST***********')
    x = 0
    y = 0
    print(save_list)
    for i in range(len(status)):
        if (status[i] == 'OK'):
            im = Image.open(save_list[i])
            name  = save_list[i].split("/")
            direct = 'static/datasets/train/OK/' + name[3] + '/' +name[4]
            print('saving to ...', direct)
            im.save(direct, 'PNG')
        else:
            im = Image.open(save_list[i])
            name  = save_list[i].split("/")
            direct2 = 'static/datasets/train/NG/' + name[3] + '/' +name[4]
            print('saving to ...', direct2)
            im.save(direct2, 'PNG')
    print('*********SAVE COMPLETED**********')
    return "nothing"


@app.route('/train')
def train():
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
    from keras.layers import Flatten, Dense, Dropout,BatchNormalization
    from keras.layers import Input, concatenate
    from keras.models import Model,load_model
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import plot_model,np_utils
    from keras import regularizers
    import keras.metrics as metric
    #train data
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True
    )
    train_generator = train_datagen.flow_from_directory(
      train_root,
      target_size=(IM_WIDTH, IM_HEIGHT),
      batch_size=batch_size,
    )

    #vaild data
    vaild_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True
    )
    vaild_generator = train_datagen.flow_from_directory(
      vaildation_root,
      target_size=(IM_WIDTH, IM_HEIGHT),
      batch_size=batch_size,
    )

    #test data
    test_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True
    )
    test_generator = train_datagen.flow_from_directory(
      test_root,
      target_size=(IM_WIDTH, IM_HEIGHT),
      batch_size=batch_size,
    )
    if os.path.exists('inception_1.h5'):
        model=load_model('inception_1.h5')
    else:
        model=check_print()

    model.fit_generator(train_generator,validation_data=vaild_generator,epochs=EPOCH,steps_per_epoch=train_generator.n/batch_size
                        ,validation_steps=vaild_generator.n/batch_size)
    model.save('inception_1.h5')
    model.metrics=['acc',metric.top_k_categorical_accuracy]
    loss,acc,top_acc=model.evaluate_generator(test_generator,steps=test_generator.n/batch_size)
    print ('Test result:loss:%f,acc:%f,top_acc:%f'%(loss,acc,top_acc))

    return "Training Complete"


@app.route('/predict')
def predict():
    global graph
    # 從參數讀取圖檔路徑
    files = []
    for filename in glob.glob('static/infer/test/NG/*.png'): #assuming jpg(can be png, gif, etc..)
        files.append(filename)
        
    result = []

    net = load_model('inception_1.h5')

    cls_list = ['NG', 'OK']

    c = 0

    # 辨識每一張圖
    for f in files:
        img = image.load_img(f, target_size=(224, 224))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        print(top_inds)
        print(f)
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
        result.append(cls_list[top_inds[0]])
        if (top_inds[0] == 1):
            c = c + 1

    ok = c
    ng = len(files) - c
    
    return render_template("predict.html", files = files, result = result, ok = c, ng = ng )
    
if __name__ == "__main__":
    app.run(port=5001,debug=True)