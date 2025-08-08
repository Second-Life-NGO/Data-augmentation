import cv2
import numpy as np

# Load the original image
image_path = ('D:/Second Life Internship/Albania/Albania Images stitched -20250617T131807Z-1-001/Albania Images '
              'stitched/Mission 1.1_compressed/stitched image M1.1.webp')  # Replace with your own file path
img = cv2.imread(image_path)

# === Enhancement Step 1: Histogram Equalization on Y channel ===
# Convert BGR to YUV color space
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# Equalize the histogram of the Y (luminance) channel
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

# Convert YUV back to BGR format
img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# === Enhancement Step 2: Gamma Correction ===
gamma = 1.2  # You can tune this value
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
img_gamma = cv2.LUT(img_eq, table)

# === Enhancement Step 3: Sharpening ===
# Define a sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Apply the kernel to the image
img_sharp = cv2.filter2D(img_gamma, -1, kernel)

# === Save the enhanced result ===
output_path = ('D:/Second Life Internship/Albania/Albania Images stitched -20250617T131807Z-1-001/Albania Images '
               'stitched/Mission 1.1_compressed/stitched image M1.1_augmented.webp')  # swap it to your own address
cv2.imwrite(output_path, img_sharp)
print(f"Enhanced image saved to: {output_path}")
