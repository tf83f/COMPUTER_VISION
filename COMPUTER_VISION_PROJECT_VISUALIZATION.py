#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature
from skimage.filters import sobel

# Load one image
img = cv2.imread('/Users/favolithomas/Desktop/CS/CV/dataverse_files/HAM10000_images_part_1/ISIC_0024366.jpg')
plt.imshow(img)
plt.title('Actual')
plt.show()

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]
plt.imshow(thresh_img, cmap='gray')
plt.title('Thresholding')
plt.show()

# Gaussian blur
blur_img = cv2.GaussianBlur(img, (5, 5), 0)
plt.imshow(blur_img[:,:,::-1])
plt.title('Gaussian Blur')
plt.show()

# Histogram equalization
eq_img = cv2.equalizeHist(gray_img)
plt.imshow(eq_img, cmap='gray')
plt.title('Histogram Equalization')
plt.show()

# Sobel edge detection
sobel_img = sobel(gray_img)
plt.imshow(sobel_img, cmap='gray')
plt.title('Sobel Edge Detection')
plt.show()

# Morphological gradient
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morph_grad = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
plt.imshow(morph_grad, cmap='gray')
plt.title('Morphological Gradient')
plt.show()

# Local Binary Patterns (LBP)
radius = 3
n_points = 8 * radius
lbp = feature.local_binary_pattern(gray_img, n_points, radius)
plt.imshow(lbp, cmap='gray')
plt.title('Local Binary Patterns (LBP)')
plt.show()

# Skin color segmentation
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 10, 60], dtype="uint8")
upper = np.array([20, 150, 255], dtype="uint8")
skin_mask = cv2.inRange(hsv_img, lower, upper)
plt.imshow(skin_mask, cmap='gray')
plt.title('Skin Color Segmentation')
plt.show()

# 3D surface plot
x, y = np.mgrid[0:gray_img.shape[0], 0:gray_img.shape[1]]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, gray_img, cmap='viridis')
ax.set_title('3D Surface Plot')
plt.show()