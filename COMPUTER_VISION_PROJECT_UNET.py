#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

# load and preprocess training/test images and masks

metadata = pd.read_csv('/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_metadata.csv')

train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=42)

train_images = []
train_masks = []

for index, row in metadata.iterrows():
    try:
        img_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/' + row['image_id'] + '.jpg'
        mask_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_segmentations_lesion_tschandl/' + row['image_id'] + '_segmentation.png'
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = img.resize((256, 256)) 
        mask = mask.resize((256, 256))
        img = np.array(img) / 255.0 
        mask = np.array(mask) / 255.0  
        mask = np.expand_dims(mask, axis=-1) 
        train_images.append(img)
        train_masks.append(mask)
    except:
        pass
    
train_images = np.array(train_images)
train_masks = np.array(train_masks)

test_images = []
test_masks = []

for index, row in test_metadata.iterrows():
    try:
        img_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/' + row['image_id'] + '.jpg'
        mask_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_segmentations_lesion_tschandl/' + row['image_id'] + '_segmentation.png'
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = img.resize((256, 256)) 
        mask = mask.resize((256, 256)) 
        img = np.array(img) / 255.0  
        mask = np.array(mask) / 255.0 
        mask = np.expand_dims(mask, axis=-1)  
        test_images.append(img)
        test_masks.append(mask)
    except:
        pass
    
test_images = np.array(test_images)
test_masks = np.array(test_masks)

train_masks = to_categorical(train_masks, num_classes=2)
test_masks = to_categorical(test_masks, num_classes=2)

# define, compile and train/test the model

inputs = Input(shape=(256, 256, 3))

convolution1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
convolution1 = Conv2D(32, (3, 3), activation='relu', padding='same')(convolution1)
pooling1 = MaxPooling2D(pool_size=(2, 2))(convolution1)

convolution2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pooling1)
convolution2 = Conv2D(64, (3, 3), activation='relu', padding='same')(convolution2)
pooling2 = MaxPooling2D(pool_size=(2, 2))(convolution2)

convolution3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pooling2)
convolution3 = Conv2D(128, (3, 3), activation='relu', padding='same')(convolution3)
pooling3 = MaxPooling2D(pool_size=(2, 2))(convolution3)

convolution4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pooling3)
convolution4 = Conv2D(256, (3, 3), activation='relu', padding='same')(convolution4)

upsampling5 = UpSampling2D(size=(2, 2))(convolution4)
upsampling5 = concatenate([upsampling5, convolution3], axis=-1)
convolution5 = Conv2D(128, (3, 3), activation='relu', padding='same')(upsampling5)
convolution5 = Conv2D(128, (3, 3), activation='relu', padding='same')(convolution5)

upsampling6 = UpSampling2D(size=(2, 2))(convolution5)
upsampling6 = concatenate([upsampling6, convolution2], axis=-1)
convolution6 = Conv2D(64, (3, 3), activation='relu', padding='same')(upsampling6)
convolution6 = Conv2D(64, (3, 3), activation='relu', padding='same')(convolution6)

upsampling7 = UpSampling2D(size=(2, 2))(convolution6)
upsampling7 = concatenate([upsampling7, convolution1], axis=-1)
convolution7 = Conv2D(32, (3, 3), activation='relu', padding='same')(upsampling7)
convolution7 = Conv2D(32, (3, 3), activation='relu', padding='same')(convolution7)

outputs = Conv2D(2, (1, 1), activation='softmax')(convolution7)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_test = model.fit(train_images, train_masks, validation_data=(test_images, test_masks), batch_size=16, epochs=10)

# plot the actual and predicted segmentation masks side by side for each image

pred_masks = model.predict(test_images)

for i in [1, 2, 3, 4]:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(test_images[i])
    ax[0].set_title('Actual Image')
    ax[0].axis('off')
    ax[1].imshow(test_masks[i][:, :, 1], cmap='gray')
    ax[1].imshow(pred_masks[i][:, :, 1], alpha=0.5, cmap='viridis')
    ax[1].set_title('Actual Mask vs. Predicted Mask')
    ax[1].axis('off')
    plt.show()
