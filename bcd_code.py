# Import library's

import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Flatten,Conv2D,MaxPooling2D
import os, cv2 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Build the data

data_path='E:/PROJECTS/research/brain_tumor_dataset'

list_folder=os.listdir() # Categories EX: ['yes','no']

# Build the data where resize (150,150)the image 

data=[]
im_size=150    # image size
for i in list_folder:
    new_path=os.path.join(data_path,i)  # Each categorie image path 
    pic_list=os.listdir(new_path)  # List of all the images in each categories 
    for img in pic_list:
        pic=os.path.join(new_path,img)   # Read the images one by one
        image_array=cv2.imread(pic,cv2.IMREAD_GRAYSCALE) # Convert it in Grayscale
        arr=cv2.resize(image_array,(im_size,im_size))   # Rescale it
        data.append([arr,list_folder.index(i)])    # Image with label

random.shuffle(data)  # Randomly shuffle the data

x_train,y_train=[],[]
for i,j in data:
    x_train.append(i)
    y_train.append(j)
x_train=np.array(x_train).reshape(-1,im_size,im_size,1)
y_train=np.array(y_train)

x=x/255

# Build the model or model structure 

model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x,y,batch_size=20,epochs=7,verbose=1)

# Now the test part
# Path where all the test images
test_data_path='E:/PROJECTS/research/brain_tumor_dataset/test'  
test_data=[]
total=os.listdir(test_data_path)
for i in total:
    test_img=os.path.join(test_data_path,i)  # Read the test images 
    test_img_arr=cv2.imread(test_img,cv2.IMREAD_GRAYSCALE)   # Read it in Grayscale
    test_arr=cv2.resize(test_img_arr,(im_size,im_size))     # Resize it
    test_arr=np.array(test_arr).reshape(-1,im_size,im_size,1)     # Convert the data type
    kk=model.predict(test_arr,batch_size=None)  # Predict the result
    test_data.append(kk[0][0])  

print(test_data) # Result
