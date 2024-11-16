# Image Gradient and Edge Detection in Image Processing
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load and preprocess image (resize and grayscale)
img = cv2.imread("Captain.jpg")
img = cv2.resize(img, (400, 300))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Laplacian Derivative for Edge Detection
lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))

# Sobel X and Y for Directional Edge Detection
sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# Combine Sobel X and Y
sobelcombine = cv2.bitwise_or(sobelX, sobelY)

# Canny Edge Detection
canny = cv2.Canny(img_gray, 100, 200)

# Thresholding to enhance edges
_, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Gradient Magnitude and Direction
gradient_magnitude = np.sqrt(sobelX**2 + sobelY**2)
gradient_magnitude = np.uint8(gradient_magnitude)
gradient_direction = np.arctan2(
    sobelY, sobelX) * (180 / np.pi)  # Convert to degrees

# Display images with OpenCV
cv2.imshow("Original", img)
cv2.imshow("Gray", img_gray)
cv2.imshow("Laplacian", lap)
cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Combined Sobel", sobelcombine)
cv2.imshow("Canny Edge Detection", canny)
cv2.imshow("Thresholded Image", thresh)
cv2.imshow("Gradient Magnitude", gradient_magnitude)

# Storing titles and images for matplotlib display
titles = ["Original", "Gray", "Laplacian", "Sobel X", "Sobel Y",
          "Combined Sobel", "Canny", "Thresholded", "Gradient Magnitude"]
images = [img, img_gray, lap, sobelX, sobelY, sobelcombine, canny,
          thresh, gradient_magnitude]

# Save the processed images
for i, image in enumerate(images):
    cv2.imwrite(f"output_{titles[i]}.jpg", image)

# Function for interactive Canny edge detection with trackbars
def update_canny(val):
    low_threshold = cv2.getTrackbarPos("Low Threshold", "Canny Edge Detection")
    high_threshold = cv2.getTrackbarPos(
        "High Threshold", "Canny Edge Detection")
    canny_updated = cv2.Canny(img_gray, low_threshold, high_threshold)
    cv2.imshow("Canny Edge Detection", canny_updated)


# window for interactive Canny adjustments
cv2.namedWindow("Canny Edge Detection")
cv2.createTrackbar("Low Threshold", "Canny Edge Detection",
                   50, 255, update_canny)
cv2.createTrackbar("High Threshold", "Canny Edge Detection",
                   150, 255, update_canny)

# Display all images with matplotlib
plt.figure(figsize=(12, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], 'gray' if i > 0 else None)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
