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

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


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
    # 資料路徑
    DATASET_PATH  = 'static/infer'

    # 影像大小
    IMAGE_SIZE = (224, 224)

    # 影像類別數
    NUM_CLASSES = 2

    # 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
    BATCH_SIZE = 8

    # 凍結網路層數
    FREEZE_LAYERS = 2

    # Epoch 數
    NUM_EPOCHS = 30

    # 模型輸出儲存的檔案
    WEIGHTS_FINAL = 'model-resnet50-final.h5'

    # 透過 data augmentation 產生訓練與驗證用的影像資料
    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       channel_shift_range=10,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                      target_size=IMAGE_SIZE,
                                                      interpolation='bicubic',
                                                      class_mode='categorical',
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE)

    valid_datagen = ImageDataGenerator()
    valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                      target_size=IMAGE_SIZE,
                                                      interpolation='bicubic',
                                                      class_mode='categorical',
                                                      shuffle=False,
                                                      batch_size=BATCH_SIZE)

    # 輸出各類別的索引值
    for cls, idx in train_batches.class_indices.items():
        print('Class #{} = {}'.format(idx, cls))

    # 以訓練好的 ResNet50 為基礎來建立模型，
    # 捨棄 ResNet50 頂層的 fully connected layers
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                   input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    x = net.output
    x = Flatten()(x)

    # 增加 DropOut layer
    x = Dropout(0.5)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    net_final.compile(optimizer=Adam(lr=1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    # 輸出整個網路結構
    print(net_final.summary())

    # 訓練模型
    net_final.fit_generator(train_batches,
                            steps_per_epoch = train_batches.samples // BATCH_SIZE,
                            validation_data = valid_batches,
                            validation_steps = valid_batches.samples // BATCH_SIZE,
                            epochs = NUM_EPOCHS)

    # 儲存訓練好的模型
    net_final.save(WEIGHTS_FINAL)

    return "Training Complete"


@app.route('/predict')
def predict():
    # 從參數讀取圖檔路徑
    files = []
    for filename in glob.glob('static/infer/test/OK/*.png'): #assuming jpg(can be png, gif, etc..)
        files.append(filename)
        
    result = []
    # 載入訓練好的模型
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