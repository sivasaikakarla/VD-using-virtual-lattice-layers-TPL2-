import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'vehicle.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Blurring the Image
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Thresholding the Image
_, thresholded = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

# Step 3: Dilating the Image
dilated = cv2.dilate(thresholded, None, iterations=3)

# Step 4: Contour Detection
contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an output image to draw contours on
contour_output = np.copy(image)
contour_output = cv2.cvtColor(contour_output, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_output, contours, -1, (0, 255, 0), 2)

# Display the results using Matplotlib
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(blurred, cmap='gray')
axs[0, 1].set_title('Blurred Image')
axs[0, 1].axis('off')

axs[1, 0].imshow(thresholded, cmap='gray')
axs[1, 0].set_title('Thresholded Image')
axs[1, 0].axis('off')

axs[1, 1].imshow(contour_output)
axs[1, 1].set_title('Contours Detected')
axs[1, 1].axis('off')

plt.show()
