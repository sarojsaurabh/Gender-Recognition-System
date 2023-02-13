from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Flatten,Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as bk
import numpy as np
from tensorflow import keras

import glob
import random
import os


import cv2

img_dim=(90,90,3)
data=[]
labels=[]


#data preprocessing
image_files=[r for r in glob.glob(r'C:\gender\Scripts\gender_dataset_face'+"/**/*",recursive=True)if not os.path.isdir(r)]
random.shuffle(image_files)
#resizing image
for img in image_files:
    image=cv2.imread(img)
    image=cv2.resize(image,(img_dim[0],img_dim[1]))
    image=img_to_array(image)
    data.append(image)

    label=img.split(os.path.sep)[-2]
    if label=="women":
        label=1
    else:
        label=0
    labels.append([label])
#pre-processing
data=np.array(data,dtype=np.uint8)/255.0 #scaling
labels=np.array(labels)

trainX,testX,trainY,testY=train_test_split(data,labels,test_size=0.2,random_state=42)

trainY=to_categorical(trainY,num_classes=2)
testY=to_categorical(testY,num_classes=2)

gen=ImageDataGenerator(rotation_range=25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

#datamodel

def build(width,height,depth,classes):
    model=Sequential()
    Shape=(height,width,depth)
    change_dim=-1

    if bk.image_data_format()=="channels_first":
        Shape=(depth,height,width)
        change_dim=1

    model.add(Conv2D(32,(3,3),padding="same",input_shape=Shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=change_dim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=change_dim))

    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=change_dim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=change_dim))

    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=change_dim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

#build model

model=build(width=img_dim[0],height=img_dim[1],depth=img_dim[2],classes=2)

b=64
epochs=100
lr=1e-3
"""
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=initial_learning_rate/epochs)
opt = keras.optimizers.SGD(learning_rate=lr_schedule)
"""
opt = Adam(learning_rate=1e-3, weight_decay=lr/epochs)

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])
H = model.fit(gen.flow(trainX, trainY,batch_size=b),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // b,
                        epochs=epochs, verbose=1)


model.save('gendermod.h5',H)

              

              

               
    
    
