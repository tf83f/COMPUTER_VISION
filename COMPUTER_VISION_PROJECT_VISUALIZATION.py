#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature
from skimage.filters import sobel

# Load one image
img = cv2.imread('/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/ISIC_0024366.jpg') # Reading an image using OpenCV
plt.imshow(img)# Displaying the image using Matplotlib
plt.title('Actual') # Setting the title for the plot
plt.show() # Displaying the plot

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting the image to grayscale using OpenCV

# Thresholding
thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1] # Applying binary thresholding to the grayscale image
plt.imshow(thresh_img, cmap='gray') # Displaying the thresholded image using Matplotlib
plt.title('Thresholding') # Setting the title for the plot
plt.show() # Displaying the plot

# Gaussian blur
blur_img = cv2.GaussianBlur(img, (5, 5), 0)  # Applying Gaussian blur to the original image
plt.imshow(blur_img[:,:,::-1]) # Displaying the blurred image using Matplotlib
plt.title('Gaussian Blur') # Setting the title for the plot
plt.show() # Displaying the plot

# Histogram equalization
eq_img = cv2.equalizeHist(gray_img) # Applying histogram equalization to the grayscale image
plt.imshow(eq_img, cmap='gray') # Displaying the equalized image using Matplotlib
plt.title('Histogram Equalization') # Setting the title for the plot
plt.show() # Displaying the plot

# Sobel edge detection
sobel_img = sobel(gray_img) # Applying the Sobel filter to the grayscale image
plt.imshow(sobel_img, cmap='gray') # Displaying the edge-detected image using Matplotlib
plt.title('Sobel Edge Detection') # Setting the title for the plot
plt.show() # Displaying the plot

# Morphological gradient
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))# Creating an elliptical structuring element for morphological gradient
morph_grad = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel) # Applying morphological gradient to the grayscale image
plt.imshow(morph_grad, cmap='gray') # Displaying the morphological gradient image using Matplotlib
plt.title('Morphological Gradient') # Setting the title for the plot
plt.show() # Displaying the plot

# Local Binary Patterns (LBP)
radius = 3 # Setting the radius for LBP
n_points = 8 * radius # Setting the number of points for LBP
lbp = feature.local_binary_pattern(gray_img, n_points, radius) # Applying LBP to the grayscale image
plt.imshow(lbp, cmap='gray')  # Displaying the LBP image using Matplotlib
plt.title('Local Binary Patterns (LBP)') # Setting the title for the plot
plt.show() # Displaying the plot

# Skin color segmentation
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV color space
lower = np.array([0, 10, 60], dtype="uint8")# Setting lower threshold values for skin color
upper = np.array([20, 150, 255], dtype="uint8") # Setting upper threshold values for skin color
skin_mask = cv2.inRange(hsv_img, lower, upper)# Creating a mask for skin color
plt.imshow(skin_mask, cmap='gray') # Displaying the skin color segmentation mask
plt.title('Skin Color Segmentation') # Setting the title for the plot
plt.show() # Displaying the plot

# 3D surface plot
x, y = np.mgrid[0:gray_img.shape[0], 0:gray_img.shape[1]] # Creating a grid of coordinates
fig = plt.figure() # Creating a figure object 
ax = fig.add_subplot(111, projection='3d')# Creating a 3D subplot
ax.plot_surface(x, y, gray_img, cmap='viridis')# Creating a 3D surface plot
ax.set_title('3D Surface Plot') # Setting the title for the plot
plt.show() # Displaying the plot
