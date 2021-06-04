import glob
import os
from sklearn.preprocessing import LabelEncoder

#from keras.utils.np_utils import *
import numpy as np
from keras import losses,models,layers,metrics,optimizers
import matplotlib.pyplot as plt
from skimage import io,transform
from sklearn.model_selection import train_test_split
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 数据预处理
# 对训练集进行处理
total_face_img = []
total_label = []
for dir_path in glob.glob(r"G:\面部年龄\facial age_datasets\face_age\face_age\*"):
    img_label = dir_path.split('/')[-1]
    label = dir_path[-3:]
    for img_path in glob.glob(os.path.join(dir_path, "*.png")):
        total_face_img.append(img_path)
        total_label.append(label)


total_image_img_list = []
for i in total_face_img:
    img = io.imread(i)
    img = transform.resize(img, (100, 100))
    img = img/255.0
    img = img.astype('float16')
    total_image_img_list.append(img)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(total_label)
#y = to_categorical(y,99)
x = np.array(total_image_img_list)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
cnn = models.Sequential()
cnn.add(layers.Conv2D(16,(3,3),activation='relu',input_shape=(100,100,3)))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.25))
cnn.add(layers.MaxPooling2D((2,2)))
cnn.add(layers.Conv2D(32,(3,3),activation='relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.25))
cnn.add(layers.MaxPooling2D((2,2)))
cnn.add(layers.Conv2D(64,(3,3),activation='relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.25))
cnn.add(layers.MaxPooling2D((2,2)))
cnn.add(layers.Flatten())
cnn.add(layers.Dropout(0.25))
cnn.add(layers.Dense(125,activation='relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dense(99,activation='softmax'))
# adam = Adam(lr=0.0001)
lr = 0.00005
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])
#cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = cnn.fit(x_train,y_train,batch_size=20,validation_data=(x_test,y_test),epochs=30,verbose=1)
# def show_history(history): #显示训练过程学习曲线
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) +1)
#     plt.figure(figsize=(12,4))
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, loss, 'bo', label='训练损失')
#     plt.plot(epochs, val_loss, 'b', label='验证损失')
#     plt.title('Training and validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, acc, 'bo', label='训练正确精度')
#     plt.plot(epochs, val_acc, 'b', label='验证正确精度')
#     plt.title('Training and validation accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()
# show_history(history)