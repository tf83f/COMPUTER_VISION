#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd # For reading csv file
import numpy as np # For working with arrays
from PIL import Image # For image processing
from sklearn.model_selection import train_test_split # For splitting data into train and test sets
from keras.utils import to_categorical # For converting labels to one-hot encoding
from keras.models import Model # For creating the model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization, Activation, Flatten, Dense # For creating the layers of the model
import matplotlib.pyplot as plt # For plotting images

# load and preprocess training/test images and masks and classes

metadata = pd.read_csv('/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_metadata.csv') # Load metadata csv file using pandas

train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=42) # Split the data into train and test sets

input_shape = (256, 256, 3) # define input shape

train_images = [] # Create an empty list to store train images
train_masks = [] # Create an empty list to store train masks
train_classes = [] # Create an empty list to store train classes

for index, row in train_metadata.iterrows():
    try:
        img_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/' + row['image_id'] + '.jpg' # Get the path of the image
        mask_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_segmentations_lesion_tschandl/' + row['image_id'] + '_segmentation.png' # Get the path of the mask
        class_data_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_metadata.csv'
        img = Image.open(img_path)  # Open the image using PIL
        mask = Image.open(mask_path) # Open the mask using PIL
        img = img.resize((256, 256)) # Resize the image to (256, 256)
        mask = mask.resize((256, 256)) # Resize the mask to (256, 256)
        img = np.array(img) / 255.0  # Normalize the pixel values of the image to [0, 1]
        mask = np.array(mask) / 255.0 # Normalize the pixel values of the mask to [0, 1]
        mask = np.expand_dims(mask, axis=-1)# Add an extra dimension to the mask
        class_data = pd.read_csv(class_data_path) # access class
        class_id = class_data[class_data['image_id'] == row['image_id']]['dx'].values[0] # add class
        train_images.append(img) # Append the image to the train_images list
        train_masks.append(mask) # Append the mask to the train_masks list
        train_classes.append(class_id) # Append the classe to the train_classes list
    except:
        pass
    
train_images = np.array(train_images) # Convert the train_images list to a numpy array
train_masks = np.array(train_masks) # Convert the train_masks list to a numpy array
train_classes = np.array(train_classes) # Convert the train_classes list to a numpy array

test_images = [] # Create an empty list to store test images
test_masks = [] # Create an empty list to store test masks
test_classes = [] # Create an empty list to store test classes

for index, row in test_metadata.iterrows():
    try:
        img_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/' + row['image_id'] + '.jpg' # Get the path of the image
        mask_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_segmentations_lesion_tschandl/' + row['image_id'] + '_segmentation.png' # Get the path of the mask
        class_data_path = '/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_metadata.csv'
        img = Image.open(img_path)  # Open the image using PIL
        mask = Image.open(mask_path) # Open the mask using PIL
        img = img.resize((256, 256)) # Resize the image to (256, 256)
        mask = mask.resize((256, 256)) # Resize the mask to (256, 256)
        img = np.array(img) / 255.0 # Normalize the pixel values of the image to [0, 1]
        mask = np.array(mask) / 255.0   # Normalize the pixel values of the mask to [0, 1]
        mask = np.expand_dims(mask, axis=-1) # Add an extra dimension to the mask
        class_data = pd.read_csv(class_data_path) # access class
        class_id = class_data[class_data['image_id'] == row['image_id']]['dx'].values[0] # add class
        test_images.append(img) # Append the image to the test_images list
        test_masks.append(mask) # Append the mask to the test_masks list
        test_classes.append(class_id) # Append the classe to the test_classes list
    except:
        pass
    
test_images = np.array(test_images) # Convert the test_images list to a numpy array
test_masks = np.array(test_masks) # Convert the test_masks list to a numpy array
test_classes = np.array(test_classes) # Convert the test_classes list to a numpy array

label_map = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6} #label_map maps class labels to integers for classification

#Convert class labels to integers using label_map
train_labels = [label_map[label] for label in train_classes]
test_labels = [label_map[label] for label in test_classes]

train_classes = to_categorical(train_labels, num_classes=7) # convert train masks to categorical
test_classes = to_categorical(test_labels, num_classes=7) # convert test masks to categorical

#Define the model architecture, compile and train/test the model

inputs = Input(shape=(256, 256, 3)) # define input shape

convolution1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)  # define first convolutional layer
convolution1 = Conv2D(64, (3, 3), activation='relu', padding='same')(convolution1) # add another convolutional layer
pooling1 = MaxPooling2D(pool_size=(2, 2))(convolution1) # add pooling layer

convolution2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pooling1) # add another convolutional layer
convolution2 = Conv2D(128, (3, 3), activation='relu', padding='same')(convolution2) # add another convolutional layer
pooling2 = MaxPooling2D(pool_size=(2, 2))(convolution2) # add pooling layer
                    
convolution3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pooling2) # add another convolutional layer
convolution3 = Conv2D(256, (3, 3), activation='relu', padding='same')(convolution3) # add another convolutional layer
convolution3 = Conv2D(256, (3, 3), activation='relu', padding='same')(convolution3) # add another convolutional layer
pooling3 = MaxPooling2D(pool_size=(2, 2))(convolution3) # add pooling layer

convolution4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pooling3) # add another convolutional layer
convolution4 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution4) # add another convolutional layer
convolution4 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution4) # add another convolutional layer
pooling4 = MaxPooling2D(pool_size=(2, 2))(convolution4) # add pooling layer

convolution5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pooling4) # add another convolutional layer
convolution5 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution5) # add another convolutional layer
convolution5 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution5) # add another convolutional layer
pooling5 = MaxPooling2D(pool_size=(2, 2))(convolution5) # add pooling layer

fullyconnected6 = Flatten()(pooling5)#Flatten the output from the previous layer to a 1D vector
fullyconnected6 = Dense(4096, activation='relu')(fullyconnected6) #Add a fully connected layer
fullyconnected6 = Dropout(0.5)(fullyconnected6)#Add dropout regularization to prevent overfitting
fullyconnected6 = Dense(4096, activation='relu')(fullyconnected6) #Add a fully connected layer
fullyconnected6 = Dropout(0.5)(fullyconnected6) #Add dropout regularization to prevent overfitting

outputs = Dense(7, activation='softmax')(fullyconnected6) # Define the output layer

model = Model(inputs, outputs) # Define the model with the input and output layers

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric

train_test = model.fit(train_images, train_classes, batch_size=16, epochs=10, validation_data=(test_images, test_classes)) #Train the model on the training data, with validation on the test data, for 10 epochs with batch size 16

# plot the actual and predicted segmentation masks side by side for each image

pred_labels = np.argmax(model.predict(test_images), axis=1) # predicts labels for test images and returns the index of the maximum value in the prediction array
test_labels = np.argmax(test_classes, axis=1) # maps the one-hot encoded test classes to their corresponding class labels

pred_labels = [list(label_map.keys())[list(label_map.values()).index(label)] for label in pred_labels] # maps the predicted labels to their corresponding class names
test_classes = [list(label_map.keys())[list(label_map.values()).index(label)] for label in test_labels] # maps the actual labels to their corresponding class names

#plotting the first test image along with its predicted mask
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(test_images[0])
ax[0].set_title('Image with actual label: ' + str(test_classes[0]))
ax[0].axis('off')
ax[1].imshow(test_masks[0], cmap='gray')
ax[1].imshow(test_images[0], alpha=0.5, cmap='viridis')
ax[1].set_title('Mask with predicted label: ' + str(pred_labels[0]))
ax[1].axis('off')
plt.show()
