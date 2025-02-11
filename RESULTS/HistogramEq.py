import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an example image
image_path = 'vehicle.png'  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Convert RGB to Grayscale
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Convert RGB to HSV
hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Extract H, S, and V channels
h_channel, s_channel, v_channel = cv2.split(hsv_image)

# Combine channels
h_plus_v = cv2.addWeighted(h_channel, 0.5, v_channel, 0.5, 0)
h_plus_s = cv2.addWeighted(h_channel, 0.5, s_channel, 0.5, 0)
s_plus_v = cv2.addWeighted(s_channel, 0.5, v_channel, 0.5, 0)
h_plus_s_plus_v = cv2.addWeighted(h_plus_s, 0.5, v_channel, 0.5, 0)

# Plot the images
plt.figure(figsize=(8, 6))

plt.subplot(3, 4, 1)
plt.imshow(image_rgb)
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(hsv_image)
plt.title('HSV Image')
plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(h_channel, cmap='hsv')
plt.title('Hue Channel')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(s_channel, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(v_channel, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(h_plus_v, cmap='gray')
plt.title('Hue + Value')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(h_plus_s, cmap='gray')
plt.title('Hue + Saturation')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(s_plus_v, cmap='gray')
plt.title('Saturation + Value')
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(h_plus_s_plus_v, cmap='gray')
plt.title('Hue + Saturation + Value')
plt.axis('off')

plt.tight_layout()
plt.show()
