#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization, Activation, Flatten, Dense
import matplotlib.pyplot as plt

# load and preprocess training/test images and masks and classes

metadata = pd.read_csv('/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_metadata.csv', nrows=1000)

train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=42)

input_shape = (256, 256, 3)

train_images = []
train_masks = []
train_classes = []

for index, row in train_metadata.iterrows():
    try:
        img_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/' + row['image_id'] + '.jpg'
        mask_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_segmentations_lesion_tschandl/' + row['image_id'] + '_segmentation.png'
        class_data_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_metadata.csv'
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = img.resize(input_shape[:2])  
        mask = mask.resize(input_shape[:2])  
        img = np.array(img) / 255.0  
        mask = np.array(mask) / 255.0  
        mask = np.expand_dims(mask, axis=-1)  
        class_data = pd.read_csv(class_data_path)
        class_id = class_data[class_data['image_id'] == row['image_id']]['dx'].values[0]
        train_images.append(img)
        train_masks.append(mask)
        train_classes.append(class_id)
    except:
        pass
    
train_images = np.array(train_images)
train_masks = np.array(train_masks)
train_classes = np.array(train_classes)

test_images = []
test_masks = []
test_classes = []

for index, row in test_metadata.iterrows():
    try:
        img_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/' + row['image_id'] + '.jpg'
        mask_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_segmentations_lesion_tschandl/' + row['image_id'] + '_segmentation.png'
        class_data_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_metadata.csv'
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = img.resize((256, 256))
        mask = mask.resize((256, 256))  
        img = np.array(img) / 255.0 
        mask = np.array(mask) / 255.0  
        mask = np.expand_dims(mask, axis=-1) 
        class_data = pd.read_csv(class_data_path)
        class_id = class_data[class_data['image_id'] == row['image_id']]['dx'].values[0]
        test_images.append(img)
        test_masks.append(mask)
        test_classes.append(class_id)
    except:
        pass
    
test_images = np.array(test_images)
test_masks = np.array(test_masks)
test_classes = np.array(test_classes)

label_map = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}

train_labels = [label_map[label] for label in train_classes]
test_labels = [label_map[label] for label in test_classes]

train_classes = to_categorical(train_labels, num_classes=7)
test_classes = to_categorical(test_labels, num_classes=7)

# define, compile and train/test the model

inputs = Input(shape=(256, 256, 3))

convolution1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
convolution1 = Conv2D(64, (3, 3), activation='relu', padding='same')(convolution1)
pooling1 = MaxPooling2D(pool_size=(2, 2))(convolution1)

convolution2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pooling1)
convolution2 = Conv2D(128, (3, 3), activation='relu', padding='same')(convolution2)
pooling2 = MaxPooling2D(pool_size=(2, 2))(convolution2)
                    
convolution3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pooling2)
convolution3 = Conv2D(256, (3, 3), activation='relu', padding='same')(convolution3)
convolution3 = Conv2D(256, (3, 3), activation='relu', padding='same')(convolution3)
pooling3 = MaxPooling2D(pool_size=(2, 2))(convolution3)

convolution4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pooling3)
convolution4 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution4)
convolution4 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution4)
pooling4 = MaxPooling2D(pool_size=(2, 2))(convolution4)

convolution5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pooling4)
convolution5 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution5)
convolution5 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution5)
pooling5 = MaxPooling2D(pool_size=(2, 2))(convolution5)

fullyconnected6 = Flatten()(pooling5)
fullyconnected6 = Dense(4096, activation='relu')(fullyconnected6)
fullyconnected6 = Dropout(0.5)(fullyconnected6)
fullyconnected6 = Dense(4096, activation='relu')(fullyconnected6)
fullyconnected6 = Dropout(0.5)(fullyconnected6)

outputs = Dense(7, activation='softmax')(fullyconnected6)

model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_test = model.fit(train_images, train_classes, batch_size=16, epochs=10, validation_data=(test_images, test_classes))

# plot the actual and predicted segmentation masks side by side for each image

pred_labels = np.argmax(model.predict(test_images), axis=1)
test_labels = np.argmax(test_classes, axis=1)

pred_labels = [list(label_map.keys())[list(label_map.values()).index(label)] for label in pred_labels]
test_classes = [list(label_map.keys())[list(label_map.values()).index(label)] for label in test_labels]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(test_images[0])
ax[0].set_title('Image with actual label: ' + str(test_classes[0]))
ax[0].axis('off')
ax[1].imshow(test_masks[0], cmap='gray')
ax[1].imshow(test_images[0], alpha=0.5, cmap='viridis')
ax[1].set_title('Mask with predicted label: ' + str(pred_labels[0]))
ax[1].axis('off')
plt.show()
