#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd # For reading csv file
import numpy as np # For working with arrays
from PIL import Image # For image processing
import matplotlib.pyplot as plt # For plotting images
from sklearn.model_selection import train_test_split # For splitting data into train and test sets
from keras.utils import to_categorical # For converting labels to one-hot encoding
from keras.models import Model # For creating the model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate # For creating the layers of the model

# load and preprocess training/test images and masks

metadata = pd.read_csv('/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_metadata.csv') # Load metadata csv file using pandas

train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=42) # Split the data into train and test sets

train_images = [] # Create an empty list to store train images
train_masks = [] # Create an empty list to store train masks

for index, row in metadata.iterrows():
    try:
        img_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/' + row['image_id'] + '.jpg' # Get the path of the image
        mask_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_segmentations_lesion_tschandl/' + row['image_id'] + '_segmentation.png' # Get the path of the mask
        img = Image.open(img_path) # Open the image using PIL
        mask = Image.open(mask_path) # Open the mask using PIL
        img = img.resize((256, 256)) # Resize the image to (256, 256)
        mask = mask.resize((256, 256)) # Resize the mask to (256, 256)
        img = np.array(img) / 255.0 # Normalize the pixel values of the image to [0, 1]
        mask = np.array(mask) / 255.0 # Normalize the pixel values of the mask to [0, 1]
        mask = np.expand_dims(mask, axis=-1)# Add an extra dimension to the mask
        train_images.append(img) # Append the image to the train_images list
        train_masks.append(mask)# Append the mask to the train_masks list
    except:
        pass
    
train_images = np.array(train_images) # Convert the train_images list to a numpy 
train_masks = np.array(train_masks) # Convert the train_masks list to a numpy array

test_images = [] # Create an empty list to store test images
test_masks = [] # Create an empty list to store test masks

for index, row in test_metadata.iterrows():
    try:
        img_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/' + row['image_id'] + '.jpg' # Get the path of the image
        mask_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_segmentations_lesion_tschandl/' + row['image_id'] + '_segmentation.png' # Get the path of the mask
        img = Image.open(img_path) # Open the image using PIL
        mask = Image.open(mask_path) # Open the mask using PIL
        img = img.resize((256, 256))  # Resize the image to (256, 256)
        mask = mask.resize((256, 256))  # Resize the mask to (256, 256)
        img = np.array(img) / 255.0 # Normalize the pixel values of the image to [0, 1]
        mask = np.array(mask) / 255.0 # Normalize the pixel values of the mask to [0, 1]
        mask = np.expand_dims(mask, axis=-1)  # Add an extra dimension to the mask
        test_images.append(img) # Append the image to the test_images list
        test_masks.append(mask) # Append the mask to the test_masks list
    except:
        pass
    
test_images = np.array(test_images) # Convert the test_images list to a numpy array
test_masks = np.array(test_masks) # Convert the test_masks list to a numpy array

train_masks = to_categorical(train_masks, num_classes=2) # convert train masks to categorical
test_masks = to_categorical(test_masks, num_classes=2) # convert test masks to categorical

#Define the model architecture, compile and train/test the model
inputs = Input(shape=(256, 256, 3)) # define input shape

convolution1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) # define first convolutional layer
convolution1 = Conv2D(32, (3, 3), activation='relu', padding='same')(convolution1) # add another convolutional layer
pooling1 = MaxPooling2D(pool_size=(2, 2))(convolution1) # add pooling layer

convolution2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pooling1) # add another convolutional layer
convolution2 = Conv2D(64, (3, 3), activation='relu', padding='same')(convolution2) # add another convolutional layer
pooling2 = MaxPooling2D(pool_size=(2, 2))(convolution2) # add pooling layer

convolution3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pooling2) # add another convolutional layer
convolution3 = Conv2D(128, (3, 3), activation='relu', padding='same')(convolution3 # add another convolutional layer
pooling3 = MaxPooling2D(pool_size=(2, 2))(convolution3) # add pooling layer

convolution4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pooling3) # add another convolutional layer
convolution4 = Conv2D(256, (3, 3), activation='relu', padding='same')(convolution4) # add another convolutional layer

upsampling5 = UpSampling2D(size=(2, 2))(convolution4) # add upsampling layer
upsampling5 = concatenate([upsampling5, convolution3], axis=-1) # concatenate with previous convolutional layer
convolution5 = Conv2D(128, (3, 3), activation='relu', padding='same')(upsampling5) # add another convolutional layer

convolution5 = Conv2D(128, (3, 3), activation='relu', padding='same')(convolution5) # add another convolutional layer

upsampling6 = UpSampling2D(size=(2, 2))(convolution5) # add upsampling layer

upsampling6 = concatenate([upsampling6, convolution2], axis=-1)# concatenate with previous convolutional layer
convolution6 = Conv2D(64, (3, 3), activation='relu', padding='same')(upsampling6) # add another convolutional layer
convolution6 = Conv2D(64, (3, 3), activation='relu', padding='same')(convolution6)# add another convolutional layer



upsampling7 = UpSampling2D(size=(2, 2))(convolution6) # add upsampling layer
upsampling7 = concatenate([upsampling7, convolution1], axis=-1)# concatenate with previous convolutional layer
convolution7 = Conv2D(32, (3, 3), activation='relu', padding='same')(upsampling7) # add another convolutional layer
convolution7 = Conv2D(32, (3, 3), activation='relu', padding='same')(convolution7) # add another convolutional layer

outputs = Conv2D(2, (1, 1), activation='softmax')(convolution7) # Define the output layer with 2 filters (one for each class) and softmax activation

model = Model(inputs=[inputs], outputs=[outputs]) # Define the model with the input and output layers

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric

train_test = model.fit(train_images, train_masks, validation_data=(test_images, test_masks), batch_size=16, epochs=10)#Train the model on the training data, with validation on the test data, for 10 epochs with batch size 16

# plot the actual and predicted segmentation masks side by side for each image

pred_masks = model.predict(test_images) #Use the trained model to predict segmentation masks for the test images

#Plot the actual and predicted segmentation masks side by side for each of the first 4 test images
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
