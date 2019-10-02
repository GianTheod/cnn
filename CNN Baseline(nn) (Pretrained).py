#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import gc

LABELS_PATH = 'C://Users//user//Desktop//pr//nn//input//height_1m//'
IMS_PATH = 'C://Users//user//Desktop/pr//nn//input//RGB_NN//'


# In[2]:


def extract_images_random(images, labels, number = 100, size = (48,48)):                                                            #Does the work of reshaping the images according to the requirement of network(48,48,3) 
    ims = np.empty((number, size[0], size[1], images.shape[2]))
    labels_map = np.empty((number, size[0], size[1]))
    labs = np.empty((number,))
    for k in range(number):
        i = np.random.randint(0, images.shape[0] - size[0])
        j = np.random.randint(0, images.shape[1] - size[1])
        ims[k, :, :, :] = images[i: i + size[0], j : j + size[1], :]/255
        labels_map[k, :, :] = labels[i: i + size[0], j : j + size[1]]
        values, counts = np.unique(labels[i: i + size[0], j : j + size[1]],return_counts = True)
        labs[k] = values[np.argmax(counts)]
    return ims, labs, labels_map


# In[3]:


import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
tqdm.pandas()

Image.MAX_IMAGE_PIXELS = None
NUMBER_PER_IMAGE = 4000
NUMBER = 8 * NUMBER_PER_IMAGE

images = []                                                                                                                       #Stores all the images converted into with the help of numpy arrays
labels = []                                                                                                                         #Stores all the corrosponding labels to the images
label_maps = []                                                                                                               #maps all the labels to the respective images 

for index, value in tqdm(enumerate(sorted(os.listdir(IMS_PATH)))):                         #This loop goes through the directory and feeds the images to the image processing function with corrosponding image labels 
    im = Image.open(IMS_PATH + value)
    im_label= Image.open(LABELS_PATH + sorted(os.listdir(LABELS_PATH))[index])
    ims, labs,labl_maps = extract_images_random(np.array(im), np.array(im_label), NUMBER_PER_IMAGE)
    
    if len(images) == 0:
        images = ims
        labels = labs
        label_maps = labl_maps
    else:
        images = np.vstack([images, ims])
        labels = np.append(labels, labs)
        label_maps = np.vstack([label_maps, labl_maps])


# In[4]:


# oversample its samples for various purposes
tiles = images
oversample = False
if oversample:
    tiles_new = tiles.copy()
    labels_new = labels.copy()
    APPENDER = 1
    TIMES = 3
    for i in tqdm(range(pd.Series(labels).value_counts()[APPENDER])):
        for i in range(TIMES):
            tiles_new = np.append(tiles_new, tiles[labels == 1][i].reshape(-1,48,48,3), axis = 0)
            labels_new = np.append(labels_new, APPENDER)
    APPENDER = 3
    TIMES = 3
    for i in tqdm(range(pd.Series(labels).value_counts()[APPENDER])):
        for i in range(TIMES):
            tiles_new = np.append(tiles_new, tiles[labels == 1][i].reshape(-1,48,48,3), axis = 0)
            labels_new = np.append(labels_new, APPENDER)

    APPENDER = 2
    TIMES = 2
    for i in tqdm(range(pd.Series(labels).value_counts()[APPENDER])):
        for i in range(TIMES):
            tiles_new = np.append(tiles_new, tiles[labels == 1][i].reshape(-1,48,48,3), axis = 0)
            labels_new = np.append(labels_new, APPENDER)
else:
    tiles_new = tiles.copy()
    labels_new = labels.copy()


# In[31]:


one_hot_labels = pd.get_dummies(labels_new).values         #Converts the labels into one hot encoding so that the images can be easily classified into different classes by the network
label_maps = label_maps.reshape(label_maps.shape[0], label_maps.shape[1], label_maps.shape[2],1)
from sklearn.model_selection import train_test_split           #Library used for splitting the data into training and testing data
tr_size = int(0.6 * tiles_new.shape[0])
val_size = int(0.2 * tiles_new.shape[0])

train_tiles, valid_tiles, test_tiles = tiles_new[:tr_size], tiles_new[tr_size:tr_size + val_size], tiles_new[tr_size + val_size:]              #Creating training testing and validation sets for the model
one_hot_train_labels, one_hot_validation_labels, one_hot_test_labels = one_hot_labels[:tr_size], one_hot_labels[tr_size:tr_size + val_size], one_hot_labels[tr_size + val_size:]   #creating corrospinding one hot encoded labels
label_maps_tr, label_maps_val, label_maps_test = label_maps[:tr_size], label_maps[tr_size:tr_size + val_size], label_maps[tr_size + val_size:]  #Mapping the corrosponding labels

# train_tiles, test_tiles, one_hot_train_labels, one_hot_test_labels = train_test_split(tiles_new, one_hot_labels)
# train_tiles, validation_tiles, one_hot_train_labels, one_hot_validation_labels = train_test_split(train_tiles, one_hot_train_labels)


# In[6]:


from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(zoom_range = 0.3,
                                     horizontal_flip = True,
                                     rotation_range = 30)                              #Creating a image generator that will feed the neural network

train_generator = train_generator.flow(train_tiles,
                                       one_hot_train_labels,
                                       batch_size = 256,
                                       shuffle = False)                                    #creating a flow pipleline that will control feeding


# In[38]:


from keras.applications import VGG16                                  #Imagenet model with state of the art accuracy and architecture .
from keras.models import Sequential, Model                       #Will be used to add custom layers over vgg
from keras.layers import Dense, Flatten, Reshape, UpSampling2D, Conv2D
from keras import optimizers, losses, metrics

def build_model():
# get layers and add average pooling layer
    conv_base = VGG16(weights='Imagenet',
    include_top=False,
    input_shape=(48, 48, 3))                                                    #Defining VGG and its parameters includ_top indicates no top most layer we will add out own also known as transfer learning.
    x = conv_base.output
    x = Flatten()(x)                                                                    #Flattening the output into 1 dimension so that more layers can be added
    x = Dense(256, activation = 'relu')(x)                                #Custom layer with 256 nodes and activation function relu(Rectified Linear unit)
    x = Dense(256, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)
    y = Reshape((8,8,1))(x)                                                     
    y = UpSampling2D((12,12))(y)
    y = Conv2D(1, kernel_size = (2,2), strides = (2,2))(y)      #Adding a convolutional layer

    # add fully-connected layer
    preds = Dense(5, activation='softmax')(x)                       #Final classifaction layer with 5 outputs indicating 5 different classes.


    model = Model(inputs=conv_base.input, outputs=[preds,y])    
    

    conv_base.trainable = True                                              #Customizing trainable nature of layers
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block4_conv1' or layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
        
    model.compile(loss=['categorical_crossentropy', 'mean_absolute_error'],
              optimizer=optimizers.RMSprop(lr=3e-5), #3e-5
              metrics=['acc'])                                                     #Compiling model with a loss function of Categorical cross entropy
    return model


# In[40]:


from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers, losses, metrics
conv_base = VGG16(weights='Imagenet',
include_top=False,
input_shape=(48, 48, 3))

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu')) # 128
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(5, activation='softmax'))

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block4_conv1' or layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
# weights = 1 - (pd.Series(labels_new).value_counts() / len(labels_new)).sort_index().values
weights = [1,4,4,4,1]
import tensorflow as tf                                              
model.compile(loss=['categorical_crossentropy'],optimizer=optimizers.RMSprop(lr=3e-5),metrics=['acc'])

model = build_model()                                                               #Building the model 

history = model.fit(x = train_tiles, y = [one_hot_train_labels, label_maps_tr], epochs=5, batch_size = 256, class_weight = [weights, weights], validation_data=(valid_tiles, [one_hot_validation_labels, label_maps_val]))#Fitting the model with all the data
print(model.evaluate(test_tiles, [one_hot_test_labels, label_maps_test]))#Testing the model with test dataset


# In[ ]:




