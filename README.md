Creating a model for image and video processing involves using libraries and frameworks that provide tools for working with visual data. One popular library for this purpose is OpenCV (Open Source Computer Vision). Additionally, you may use deep learning frameworks such as TensorFlow or PyTorch for more advanced tasks like object detection or image classification. Below, I'll provide a basic example using OpenCV for image processing and a simple image classification model using TensorFlow/Keras.

### Image Processing with OpenCV:

```python
import cv2
import numpy as np

# Load an image from file
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred_image, 50, 150)

# Display the original and processed images
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Simple Image Classification with TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

# Load a pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Load and preprocess an image for classification
image_path = 'path/to/your/image.jpg'
img = image.load_img(image_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Decode and print the top-3 predicted classes
decoded_predictions = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

# You can further process the predictions based on your needs
```

Make sure to install the necessary libraries before running these scripts. You can install them using:

```bash
pip install opencv-python tensorflow
```

Feel free to adapt these examples based on your specific requirements and use case. Image and video processing can involve various techniques, and the choice of model or method depends on the specific task you want to accomplish.

Certainly! Let's extend the example with more advanced image processing using OpenCV and then move on to video processing.

### Advanced Image Processing with OpenCV:

```python
import cv2
import numpy as np

# Load an image from file
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the original, processed, and contoured images
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This script adds contour detection to the previous example. It finds contours in the edged image and draws them on the original image.

### Video Processing with OpenCV:

```python
import cv2

# Open a video capture object
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read and process frames from the video
while True:
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Perform some processing on each frame (e.g., convert to grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the original and processed frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Grayscale Frame', gray_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

This script reads frames from a video file, converts each frame to grayscale, and displays both the original and processed frames. Press the 'q' key to exit the video playback.

Feel free to experiment with these examples and adapt them to your specific use case or project requirements. Image and video processing offer a wide range of possibilities, and OpenCV provides a powerful set of tools to work with visual data in Python.

Certainly! Let's continue with more advanced topics, such as object detection using a pre-trained deep learning model and real-time video processing.

### Object Detection with OpenCV and a Pre-trained Model (Using MobileNet SSD):

```python
import cv2

# Load pre-trained MobileNet SSD model and its configuration
net = cv2.dnn.readNetFromCaffe('path/to/MobileNetSSD_deploy.prototxt', 'path/to/MobileNetSSD_deploy.caffemodel')

# Open a video capture object
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the class labels for MobileNet SSD
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Read and process frames from the video
while True:
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Perform object detection using MobileNet SSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:  # Filter out weak detections
            class_id = int(detections[0, 0, i, 1])
            label = f"{classes[class_id]}: {confidence:.2f}%"

            # Draw bounding box and label on the frame
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with object detections
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

Make sure to download the MobileNet SSD model files (`MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel`) from the [official OpenCV GitHub repository](https://github.com/chuanqi305/MobileNet-SSD) and provide the correct paths in the script.

### Real-time Face Detection with OpenCV and Haarcascades:

```python
import cv2

# Load the pre-trained Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video capture object
cap = cv2.VideoCapture(0)  # Use the default camera (you can change the parameter to a video file path)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Read and process frames from the camera
while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with face detections
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

These examples cover a range of topics from object detection using a pre-trained model to real-time face detection. Feel free to modify and combine these scripts based on your specific needs and project requirements.

Certainly! Let's explore a few more advanced concepts, such as image segmentation and background subtraction.

### Image Segmentation with OpenCV (Using GrabCut):

```python
import cv2
import numpy as np

# Load an image from file
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)

# Create a mask (1 for foreground, 2 for probable foreground, 0 for background, 3 for probable background)
mask = np.zeros(image.shape[:2], np.uint8)

# Define the region of interest (ROI) for segmentation
rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to create a binary mask (0 and 2 for background, 1 and 3 for foreground)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiply the image with the binary mask to get the segmented image
result = image * mask2[:, :, np.newaxis]

# Display the original and segmented images
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This script uses GrabCut, an interactive segmentation algorithm, to segment an object from the background. You can manually adjust the rectangle coordinates to include the object you want to segment.

### Background Subtraction for Moving Object Detection:

```python
import cv2
import numpy as np

# Open a video capture object
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Read and process frames from the video
while True:
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Display the original frame and the foreground mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

This script uses the MOG2 (Mixture of Gaussians) algorithm for background subtraction to detect moving objects in a video.

Feel free to experiment with these examples and integrate them into your projects. Advanced image and video processing often involve a combination of these techniques to achieve specific goals, such as object recognition, segmentation, and motion detection.

Certainly! Let's explore additional concepts, such as optical flow for motion tracking and using a pre-trained deep learning model for image segmentation.

### Optical Flow for Motion Tracking:

```python
import cv2
import numpy as np

# Open a video capture object
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame1 = cap.read()

# Convert the first frame to grayscale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the current frame to grayscale
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert polar coordinates to Cartesian
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for visualization
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Display the optical flow
    cv2.imshow('Optical Flow', rgb)

    # Update the previous frame
    prvs = next_frame

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

This script uses the Farneback method for optical flow to estimate motion between consecutive frames in a video.

### Image Segmentation with a Pre-trained Deep Learning Model (Using DeepLabv3+):

```python
import cv2
import numpy as np
from PIL import Image

# Load a pre-trained DeepLabv3+ model
net = cv2.dnn.readNetFromTensorflow('path/to/deeplabv3_257_mv_gpu.tflite')

# Open an image from file
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)

# Resize the image to match the input size of the model
input_size = (257, 257)
input_image = cv2.resize(image, input_size)

# Prepare the input image for the model
input_blob = cv2.dnn.blobFromImage(input_image, scalefactor=1.0 / 127.5, mean=[127.5, 127.5, 127.5], swapRB=True)

# Set the input to the model
net.setInput(input_blob)

# Run inference
output = net.forward()

# Get the segmentation mask
mask = output[0, 0]

# Threshold the mask to create a binary mask
_, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

# Resize the binary mask to the original image size
binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

# Apply the binary mask to the original image
segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)

# Display the original image, segmentation mask, and segmented image
cv2.imshow('Original Image', image)
cv2.imshow('Segmentation Mask', binary_mask)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This script uses a pre-trained DeepLabv3+ model to perform semantic segmentation on an image.

Make sure to download the necessary model files and provide the correct paths in the script. You can find the DeepLabv3+ model files on the [official TensorFlow GitHub repository](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md).

Feel free to explore and combine these examples based on your specific requirements and project goals. Advanced image and video processing often involve a mix of traditional computer vision techniques and deep learning approaches.

Certainly! Let's explore some more advanced concepts, including facial recognition and pose estimation.

### Facial Recognition with OpenCV and dlib:

First, you'll need to install the `dlib` library. You can do this by running:

```bash
pip install dlib
```

Now, you can use the following script for facial recognition:

```python
import cv2
import dlib

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmarks predictor from dlib
predictor = dlib.shape_predictor('path/to/shape_predictor_68_face_landmarks.dat')

# Open a video capture object
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray_frame)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray_frame, face)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Draw facial landmarks
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Display the frame with face detection and landmarks
    cv2.imshow('Facial Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

This script uses the dlib library for face detection and facial landmarks prediction. Make sure to download the pre-trained facial landmarks predictor (`shape_predictor_68_face_landmarks.dat`) from the [dlib model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and provide the correct path in the script.

### Pose Estimation with OpenCV and PoseNet:

For pose estimation, you can use the PoseNet model, which is available in the `cv2` module:

```python
import cv2

# Open a video capture object
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load the pre-trained PoseNet model
net = cv2.dnn.readNetFromTensorflow('path/to/pose/pose_iter_440000.caffemodel', 'path/to/pose/pose_deploy_linevec.prototxt')

while True:
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Resize the frame to match the input size of the model
    input_size = (368, 368)
    input_frame = cv2.resize(frame, input_size)

    # Prepare the input frame for the model
    input_blob = cv2.dnn.blobFromImage(input_frame, 1.0 / 255, input_size, (127.5, 127.5, 127.5), swapRB=True, crop=False)

    # Set the input to the model
    net.setInput(input_blob)

    # Run inference
    output = net.forward()

    # Draw the keypoints and skeletal structure on the frame
    points = output[0, 0, :, :]
    for i in range(points.shape[0]):
        x = int(points[i, 0] * frame.shape[1])
        y = int(points[i, 1] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    # Display the frame with pose estimation
    cv2.imshow('Pose Estimation', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

Make sure to download the pre-trained PoseNet model files (`pose_iter_440000.caffemodel` and `pose_deploy_linevec.prototxt`) from the [official PoseNet GitHub repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_iter_440000.caffemodel) and provide the correct paths in the script.

Feel free to experiment with these examples and adapt them to your specific use case or project requirements. Advanced image and video processing involve a combination of techniques and models based on the specific tasks you want to achieve.

Certainly! Let's explore a few more advanced concepts, including image stitching and object tracking.

### Image Stitching with OpenCV (Panorama Creation):

```python
import cv2
import numpy as np

# Read a series of images for stitching
image1 = cv2.imread('path/to/image1.jpg')
image2 = cv2.imread('path/to/image2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect key points and descriptors using ORB
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Use a brute-force matcher to find the best matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort the matches based on their distances
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched key points
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find the homography matrix
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

# Warp the first image to align with the second image
result = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))

# Overlay the second image onto the result
result[0:image2.shape[0], 0:image2.shape[1]] = image2

# Display the original images and the stitched panorama
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.imshow('Stitched Panorama', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This script demonstrates image stitching using the ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor.

### Object Tracking with OpenCV (Using Mean Shift):

```python
import cv2
import numpy as np

# Open a video capture object
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()

# Define the region of interest (ROI) for tracking
x, y, w, h = 200, 150, 100, 100
roi = frame[y:y + h, x:x + w]

# Convert the ROI to the HSV color space
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create a mask and histogram for the ROI
roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Set the termination criteria for mean shift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to the HSV color space
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the back projection of the frame using the ROI histogram
    dst = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)

    # Apply mean shift to track the object
    ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)

    # Draw the tracked object on the frame
    x, y, w, h = track_window
    img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

    # Display the frame with the tracked object
    cv2.imshow('Object Tracking', img2)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

This script demonstrates object tracking using the Mean Shift algorithm. The initial region of interest (ROI) is defined, and Mean Shift is applied to track the object in subsequent frames.

Feel free to adapt these examples based on your specific requirements and use cases. Advanced image and video processing often involve combining various techniques and algorithms to achieve specific tasks.

Certainly! Let's explore a couple more advanced concepts: image inpainting and real-time emotion detection.

### Image Inpainting with OpenCV:

Image inpainting is the process of filling in missing or damaged parts of an image. OpenCV provides an inpainting algorithm that can be used for such tasks. Here's a basic example:

```python
import cv2
import numpy as np

# Read an image with a region to be inpainted
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)
mask = cv2.imread('path/to/your/mask.jpg', cv2.IMREAD_GRAYSCALE)

# Perform inpainting
inpainting = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Display the original image, mask, and inpainted result
cv2.imshow('Original Image', image)
cv2.imshow('Mask', mask)
cv2.imshow('Inpainting Result', inpainting)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Make sure to create a mask image where the region to be inpainted is marked with white pixels and save it as `mask.jpg`.

### Real-time Emotion Detection with OpenCV and a Pre-trained Model:

For real-time emotion detection, you can use a pre-trained deep learning model. Here, we'll use the Haarcascades for face detection and a pre-trained emotion detection model:

```python
import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained emotion detection model
model = load_model('path/to/emotion_model.h5')

# Open a video capture object
cap = cv2.VideoCapture(0)  # Use the default camera

# Load the pre-trained Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the pixel values
        face_roi = face_roi / 255.0

        # Reshape the input for the model
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Predict emotion
        emotion_prediction = model.predict(face_roi)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with face detection and emotion prediction
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

Make sure to download the emotion detection model (e.g., Fer2013_mini_XCEPTION.102-0.66.hdf5) from the [Fer2013 Kaggle dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) or train your own model.

Feel free to experiment with these examples and adapt them to your specific use case or project requirements. Advanced image and video processing often involve a mix of traditional computer vision techniques and deep learning approaches.

Certainly! Let's explore a couple more advanced concepts: text recognition and style transfer.

### Text Recognition with Tesseract OCR and OpenCV:

Tesseract OCR is an open-source tool for recognizing text in images. You can use the `pytesseract` wrapper along with OpenCV for text recognition. Ensure you have Tesseract installed on your system:

```bash
sudo apt-get install tesseract-ocr
pip install pytesseract
```

Now, you can perform text recognition using the following script:

```python
import cv2
import pytesseract

# Set the path to the Tesseract executable (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Read an image with text
image_path = 'path/to/your/image_with_text.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use pytesseract to perform OCR
text = pytesseract.image_to_string(gray)

# Display the original image and recognized text
cv2.imshow('Original Image', image)
print(f"Recognized Text:\n{text}")

cv2.waitKey(0)
cv2.destroyAllWindows()
```

This script reads an image with text, converts it to grayscale, and uses Tesseract OCR to extract the text content.

### Style Transfer with Neural Style Transfer Algorithm:

Neural Style Transfer allows you to apply the artistic style of one image to the content of another. Here's a simple example using the VGG19 neural network and the `keras-contrib` library:

```bash
pip install keras tensorflow numpy
```

Now, you can create a script for style transfer:

```python
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
from keras_contrib.applications import neural_transfer

# Set the paths to your content and style images
content_image_path = 'path/to/your/content_image.jpg'
style_image_path = 'path/to/your/style_image.jpg'

# Load content and style images
content_image = load_img(content_image_path, target_size=(400, 400))
style_image = load_img(style_image_path, target_size=(400, 400))

content_array = img_to_array(content_image)
content_array = np.expand_dims(content_array, axis=0)
style_array = img_to_array(style_image)
style_array = np.expand_dims(style_array, axis=0)

# Preprocess input for VGG19 model
content_array = vgg19.preprocess_input(content_array)
style_array = vgg19.preprocess_input(style_array)

# Define content layer and style layers for neural transfer
content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Build the neural transfer model
model = neural_transfer.build_model(content_array, style_array, content_layer, style_layers)

# Run neural transfer
generated_image = neural_transfer.run_transfer(model, content_array, style_array, iterations=100)

# Convert the generated image back to a displayable format
generated_image = neural_transfer.deprocess_image(generated_image)

# Display the content, style, and generated images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(load_img(content_image_path))
plt.title('Content Image')

plt.subplot(1, 3, 2)
plt.imshow(load_img(style_image_path))
plt.title('Style Image')

plt.subplot(1, 3, 3)
plt.imshow(generated_image)
plt.title('Generated Image')

plt.show()
```

This script uses the `keras-contrib` library to perform neural style transfer. Make sure to adjust the paths to your content and style images.

Feel free to experiment with these examples and adapt them to your specific use cases or project requirements. Advanced image and video processing often involve combining various techniques and algorithms to achieve specific tasks.

Certainly! Let's explore another advanced concept: image super-resolution using deep learning.

### Image Super-Resolution with OpenCV and Deep Learning (Using ESRGAN):

Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) is a deep learning model for image super-resolution. You can use the `cv2` and `requests` libraries to download and apply a pre-trained ESRGAN model. First, install the required libraries:

```bash
pip install opencv-python requests
```

Now, create a script for image super-resolution:

```python
import cv2
import numpy as np
import requests
from io import BytesIO

# Function to download and load the ESRGAN model
def load_esrgan_model():
    model_url = 'https://github.com/xinntao/ESRGAN/releases/download/1.2/ESRGAN_x4.pth'
    model_weights = requests.get(model_url).content
    return cv2.dnn.readNetFromTorch(BytesIO(model_weights))

# Function for image super-resolution using ESRGAN
def super_resolve_image(image, esrgan_model):
    # Resize the image to a multiple of 4 for better compatibility
    h, w, _ = image.shape
    new_h = (h // 4) * 4
    new_w = (w // 4) * 4
    image = cv2.resize(image, (new_w, new_h))

    # Prepare the image for ESRGAN
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(new_w, new_h))
  
    # Set the input to the model
    esrgan_model.setInput(blob)

    # Run the model to get the super-resolved image
    output = esrgan_model.forward()
  
    # Reshape and normalize the output
    output = output.squeeze().transpose(1, 2, 0)
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    return output

# Load the ESRGAN model
esrgan_model = load_esrgan_model()

# Read an image for super-resolution
image_path = 'path/to/your/image.jpg'
input_image = cv2.imread(image_path)

# Perform image super-resolution
super_res_image = super_resolve_image(input_image, esrgan_model)

# Display the original and super-resolved images
cv2.imshow('Original Image', input_image)
cv2.imshow('Super-Resolved Image', super_res_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This script downloads a pre-trained ESRGAN model and uses it to perform image super-resolution on the input image.

Feel free to experiment with these examples and adapt them to your specific use cases or project requirements. Advanced image processing often involves combining various techniques and leveraging deep learning models to achieve specific tasks.

Certainly! Let's explore one more advanced concept: image segmentation using a deep learning model, specifically the U-Net architecture.

### Image Segmentation with U-Net Architecture:

U-Net is a convolutional neural network architecture commonly used for image segmentation tasks. You can use a pre-trained U-Net model to perform semantic segmentation on images. In this example, I'll use the `segmentation-models` library, which provides pre-trained models for image segmentation tasks.

```bash
pip install keras numpy matplotlib
pip install segmentation-models
```

Now, create a script for image segmentation using U-Net:

```python
import cv2
import numpy as np
from keras.models import load_model
from segmentation_models import Unet
from segmentation_models import get_preprocessing

# Function to preprocess input image for the U-Net model
def preprocess_input(image):
    return get_preprocessing('resnet34')(image)

# Load a pre-trained U-Net model
model = Unet('resnet34', input_shape=(None, None, 3), classes=1, activation='sigmoid')
model.load_weights('path/to/unet_model_weights.h5')  # Provide the path to your pre-trained weights

# Read an image for segmentation
image_path = 'path/to/your/image.jpg'
input_image = cv2.imread(image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Preprocess the input image
preprocessed_input = preprocess_input(input_image)
preprocessed_input = np.expand_dims(preprocessed_input, axis=0)

# Perform image segmentation using the U-Net model
segmentation_mask = model.predict(preprocessed_input)[0]

# Threshold the segmentation mask to create a binary mask
binary_mask = (segmentation_mask > 0.5).astype(np.uint8)

# Apply the binary mask to the original image
segmented_image = cv2.bitwise_and(input_image, input_image, mask=binary_mask)

# Display the original image, segmentation mask, and segmented image
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(segmentation_mask.squeeze(), cmap='gray')
plt.title('Segmentation Mask')

plt.subplot(1, 3, 3)
plt.imshow(segmented_image)
plt.title('Segmented Image')

plt.show()
```

In this script, a pre-trained U-Net model is loaded, and an input image is processed to generate a segmentation mask. The mask is thresholded to create a binary mask, and the original image is segmented based on this mask.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Advanced image processing often involves a combination of traditional computer vision techniques and deep learning approaches to achieve specific goals.

Certainly! Let's explore the concept of image style transfer using deep neural networks.

### Image Style Transfer with Neural Style Transfer:

Neural Style Transfer is a technique that allows you to apply the artistic style of one image (the "style image") to the content of another image (the "content image"). Here, we'll use a deep neural network to achieve this effect.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image style transfer:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return tf.convert_to_tensor(image)

# Function to deprocess and display an image
def deprocess_and_display_image(image):
    image = image.numpy()
    image = image.reshape(image.shape[1:])  # Remove batch dimension
    image = tf.keras.applications.vgg19.decode_predictions(image, top=1)[0][0]
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Load the VGG19 model pre-trained on ImageNet (excluding the fully connected layers)
vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False)

# Choose a content image and a style image
content_image_path = 'path/to/your/content_image.jpg'
style_image_path = 'path/to/your/style_image.jpg'

# Load and preprocess the content and style images
content_image = load_and_preprocess_image(content_image_path)
style_image = load_and_preprocess_image(style_image_path)

# Visualize the content and style images
deprocess_and_display_image(content_image)
deprocess_and_display_image(style_image)

# Extract feature maps from intermediate layers for both content and style images
content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

content_model = tf.keras.models.Model(inputs=vgg19.input, outputs=[vgg19.get_layer(layer).output for layer in content_layers])
style_model = tf.keras.models.Model(inputs=vgg19.input, outputs=[vgg19.get_layer(layer).output for layer in style_layers])

content_features = content_model(content_image)
style_features = style_model(style_image)

# Define a function to compute the Gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)

# Compute Gram matrices for style features
style_gram_matrices = [gram_matrix(style_feature) for style_feature in style_features]

# Build the Neural Style Transfer model
class StyleTransferModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleTransferModel, self).__init__()
        self.vgg = vgg19
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

# Define the target content and style weights
style_weight = 1e-2
content_weight = 1e4

# Create the StyleTransferModel instance
model = StyleTransferModel(style_layers, content_layers)

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Create a function to compute the total loss
def total_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    # Compute the output of the model
    outputs = model(init_image)

    # Extract style and content outputs
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    # Initialize the total loss
    total_style_loss = tf.zeros_like(style_outputs['block1_conv1'])
    total_content_loss = tf.zeros_like(content_outputs['block4_conv2'])

    # Compute the style loss for each style layer
    for style_layer in style_outputs.keys():
        style_output = style_outputs[style_layer]
        style_loss = style_loss_function(style_output, gram_style_features[style_layer])
        total_style_loss += style_loss / len(style_layers)

    # Compute the content loss for each content layer
    for content_layer in content_outputs.keys():
        content_output = content_outputs[content_layer]
        content_loss = content_loss_function(content_output, content_features[content_layer])
        total_content_loss += content_loss / len(content_layers)

    # Compute the total loss
    total_loss = (style_weight * total_style_loss +

 content_weight * total_content_loss)

    return total_loss

# Create a function to compute the gradients and apply them to the optimizer
def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        loss = total_loss(**cfg)
    grads = tape.gradient(loss, cfg['init_image'])
    optimizer.apply_gradients([(grads, cfg['init_image'])])
    return loss, grads

# Define the style loss function
def style_loss_function(style, gram_style):
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram_style))

# Define the content loss function
def content_loss_function(content, target):
    return tf.reduce_mean(tf.square(content - target))

# Initialize the target image for style transfer
init_image = content_image

# Define the number of iterations and the step size
num_iterations = 1000
step_size = 10

# Store the best result
best_loss, best_img = float('inf'), None

# Compute the Gram matrices for style features
gram_style_features = {style_layer: gram_matrix(style_features[style_layer]) for style_layer in style_features.keys()}

# Create a configuration dictionary for computing gradients
cfg = {
    'model': model,
    'loss_weights': (style_weight, content_weight),
    'init_image': init_image,
    'gram_style_features': gram_style_features,
    'content_features': content_features
}

# Perform style transfer
for i in range(num_iterations):
    loss, grads = compute_grads(cfg)
  
    # Update the target image using the computed gradients
    init_image = init_image + step_size * grads

    # Clip the pixel values to be in the valid range [0, 255]
    init_image = tf.clip_by_value(init_image, clip_value_min=0.0, clip_value_max=255.0)

    # Print the loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.numpy()}")

    # Update the best result if a lower loss is found
    if loss < best_loss:
        best_loss = loss
        best_img = tf.image.convert_image_dtype(init_image[0], tf.uint8)

# Display the final stylized image
deprocess_and_display_image(best_img)
```

This script uses the VGG19 neural network to extract feature maps from both the style and content images. It then defines a loss function that measures the difference between the Gram matrices of the feature maps. The script iteratively adjusts the pixel values of the target image to minimize this loss, resulting in a stylized image.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Advanced image processing often involves a combination of traditional computer vision techniques and deep learning approaches to achieve specific goals.

Certainly! Let's explore another advanced concept: image anomaly detection using autoencoders.

### Image Anomaly Detection with Autoencoders:

Autoencoders are neural network architectures used for unsupervised learning. In the context of image anomaly detection, an autoencoder is trained on a dataset of normal images, and then it is used to reconstruct test images. Anomalies can be detected by measuring the difference between the original and reconstructed images.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image anomaly detection using autoencoders:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(64, 64)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    return image

# Load and preprocess normal images for training the autoencoder
normal_image_paths = ['path/to/normal_image1.jpg', 'path/to/normal_image2.jpg', ...]
normal_images = [load_and_preprocess_image(path) for path in normal_image_paths]

# Convert the list of normal images to a NumPy array
normal_images = np.array(normal_images)

# Define the architecture of the autoencoder
latent_dim = 64

encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(latent_dim, activation='relu')
])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
    tf.keras.layers.Dense(16 * 16 * 64, activation='relu'),
    tf.keras.layers.Reshape((16, 16, 64)),
    tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
])

autoencoder = tf.keras.Sequential([encoder, decoder])

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder on normal images
autoencoder.fit(normal_images, normal_images, epochs=50, batch_size=32)

# Function to load and preprocess a test image
def load_and_preprocess_test_image(image_path, target_size=(64, 64)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    return image

# Load and preprocess a test image with potential anomalies
test_image_path = 'path/to/test_image.jpg'
test_image = load_and_preprocess_test_image(test_image_path)

# Expand dimensions to match the expected input shape for the autoencoder
test_image = np.expand_dims(test_image, axis=0)

# Use the trained autoencoder to reconstruct the test image
reconstructed_image = autoencoder.predict(test_image)

# Calculate the mean squared error (MSE) between the original and reconstructed images
mse = np.mean(np.square(test_image - reconstructed_image))

# Define a threshold for anomaly detection (adjust as needed)
threshold = 0.02

# Display the original and reconstructed images, and indicate if it's an anomaly
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(test_image[0])
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image[0])
plt.title(f'Reconstructed Image\nMSE: {mse:.4f}')

plt.show()

# Print whether the test image is considered an anomaly
if mse > threshold:
    print("Anomaly Detected!")
else:
    print("No Anomaly Detected.")
```

In this script, an autoencoder is trained on a dataset of normal images, and then it is used to reconstruct a test image. The Mean Squared Error (MSE) between the original and reconstructed images is calculated. If the MSE exceeds a predefined threshold, the test image is considered an anomaly.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Advanced image processing often involves a combination of traditional computer vision techniques and deep learning approaches to achieve specific goals.

Certainly! Let's explore another advanced concept: image captioning using a combination of Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) for generating captions.

### Image Captioning with CNNs and RNNs:

Image captioning involves generating textual descriptions for images. In this example, we'll use a pre-trained CNN to extract image features and an LSTM-based RNN to generate captions.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image captioning:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the InceptionV3 model pre-trained on ImageNet (excluding the fully connected layers)
inception_model = InceptionV3(weights='imagenet')
inception_model = tf.keras.Model(inception_model.input, inception_model.layers[-2].output)

# Function to preprocess and extract features from an image
def extract_features(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = inception_model.predict(img_array)
    return features

# Load and preprocess an example image
example_image_path = 'path/to/example_image.jpg'
example_features = extract_features(example_image_path)

# Display the example image
plt.imshow(load_img(example_image_path))
plt.title('Example Image')
plt.show()

# Load a pre-trained tokenizer and model for generating captions
tokenizer = Tokenizer()
tokenizer.word_index = np.load('path/to/tokenizer_word_index.npy', allow_pickle=True).item()

caption_model = load_model('path/to/caption_model.h5')

# Function to generate a caption for an image
def generate_caption(image_features, max_len=20):
    input_text = 'startseq'
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)
        predicted_word_index = np.argmax(caption_model.predict([image_features, sequence]))
        predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_word_index][0]
        if predicted_word == 'endseq':
            break
        input_text += ' ' + predicted_word
    return input_text.replace('startseq', '').replace('endseq', '').strip()

# Generate and display a caption for the example image
generated_caption = generate_caption(example_features)
print('Generated Caption:', generated_caption)

# Display the example image with the generated caption
plt.imshow(load_img(example_image_path))
plt.title('Generated Caption: ' + generated_caption)
plt.show()
```

In this script, the InceptionV3 model is used to extract features from an example image. These features are then used as input to the pre-trained model for image captioning. The model generates a caption for the example image.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Image captioning is a fascinating area that combines computer vision and natural language processing to describe the content of images in a human-readable way.

Certainly! Let's explore another advanced concept: image segmentation using a more sophisticated model called DeepLabV3.

### Image Segmentation with DeepLabV3:

DeepLabV3 is a state-of-the-art deep learning model for semantic image segmentation. It utilizes a deep convolutional neural network to assign semantic labels to each pixel in an image. In this example, we'll use a pre-trained DeepLabV3 model to perform image segmentation.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image segmentation using DeepLabV3:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.deeplab_v3 import preprocess_input, decode_predictions

# Load the DeepLabV3 model pre-trained on PASCAL VOC dataset
deeplab_model = tf.keras.applications.DeepLabV3(weights='pascal_voc')

# Function to preprocess and predict segmentation mask for an image
def predict_segmentation(image_path):
    img = load_img(image_path, target_size=(512, 512))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    segmentation_mask = deeplab_model.predict(img_array)
    return segmentation_mask

# Load and preprocess an example image for segmentation
example_image_path = 'path/to/example_image.jpg'
segmentation_mask = predict_segmentation(example_image_path)

# Display the original image and the predicted segmentation mask
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(load_img(example_image_path))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmentation_mask.squeeze())
plt.title('Segmentation Mask')

plt.show()
```

In this script, the DeepLabV3 model is used to predict a segmentation mask for an example image. The model has been pre-trained on the PASCAL VOC dataset, which includes various object categories.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Image segmentation is a crucial task in computer vision, and models like DeepLabV3 are powerful tools for understanding the content and context of images at the pixel level.

Certainly! Let's explore another advanced concept: object detection using a popular framework called TensorFlow Object Detection API.

### Object Detection with TensorFlow Object Detection API:

TensorFlow Object Detection API is a powerful framework for building and deploying object detection models. In this example, we'll use a pre-trained model from the API to perform object detection on an image.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for object detection using TensorFlow Object Detection API:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

# Load the pre-trained model and label map
model_path = 'path/to/ssd_inception_v2_coco/saved_model'
label_map_path = 'path/to/mscoco_label_map.pbtxt'

model = tf.saved_model.load(model_path)
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Function to perform object detection on an image
def perform_object_detection(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor([image_np], dtype=tf.uint8)

    detections = model(input_tensor)
  
    # Visualize the detections on the image
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(detections['detection_boxes']),
        np.squeeze(detections['detection_classes']).astype(np.int32),
        np.squeeze(detections['detection_scores']),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8
    )

    return image_np

# Load and perform object detection on an example image
example_image_path = 'path/to/example_image.jpg'
result_image = perform_object_detection(example_image_path)

# Display the original and result images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(Image.open(example_image_path))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(result_image)
plt.title('Object Detection Result')

plt.show()
```

In this script, the TensorFlow Object Detection API is used to load a pre-trained model (SSD with Inception V2 trained on COCO dataset) and a label map. The model is then used to perform object detection on an example image, and the results are visualized.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Object detection is a fundamental task in computer vision, and frameworks like TensorFlow Object Detection API make it easier to work with pre-trained models for various applications.

Certainly! Let's explore another advanced concept: image style transfer using Generative Adversarial Networks (GANs).

### Image Style Transfer with GANs:

Generative Adversarial Networks (GANs) can be used for image style transfer by learning to generate images that combine the content of one image with the style of another. In this example, we'll use a pre-trained GAN model for image style transfer.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image style transfer using GANs:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained GAN model
gan_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)

# Function to perform image style transfer
def perform_style_transfer(content_image_path, style_image_path):
    # Load and preprocess content and style images
    content_image = load_img(content_image_path, target_size=(224, 224))
    content_image = img_to_array(content_image)
    content_image = np.expand_dims(content_image, axis=0)
    content_image = tf.keras.applications.efficientnet.preprocess_input(content_image)

    style_image = load_img(style_image_path, target_size=(224, 224))
    style_image = img_to_array(style_image)
    style_image = np.expand_dims(style_image, axis=0)
    style_image = tf.keras.applications.efficientnet.preprocess_input(style_image)

    # Generate the stylized image
    generated_image = gan_model.predict(content_image + style_image)

    # Clip values to be in the valid image range
    generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)

    return generated_image[0]

# Load and perform style transfer on an example image
content_image_path = 'path/to/content_image.jpg'
style_image_path = 'path/to/style_image.jpg'
result_image = perform_style_transfer(content_image_path, style_image_path)

# Display the original content, style, and result images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(load_img(content_image_path))
plt.title('Content Image')

plt.subplot(1, 3, 2)
plt.imshow(load_img(style_image_path))
plt.title('Style Image')

plt.subplot(1, 3, 3)
plt.imshow(result_image)
plt.title('Stylized Image')

plt.show()
```

In this script, an EfficientNetB0 model pre-trained on ImageNet is used for image style transfer. The content image provides the structure, and the style image contributes artistic elements to generate a stylized image.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Image style transfer using GANs is an exciting application that allows you to create visually appealing and artistic images.

Certainly! Let's explore another advanced concept: image inpainting using deep learning.

### Image Inpainting with Deep Learning:

Image inpainting involves filling in missing or damaged parts of an image. Deep learning models can be trained to understand the context of an image and generate plausible content to fill in the missing regions. In this example, we'll use a pre-trained deep learning model for image inpainting.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image inpainting using a pre-trained model:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained inpainting model
inpainting_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False)

# Function to perform image inpainting
def perform_inpainting(image_path, mask_path):
    # Load and preprocess the image and mask
    image = load_img(image_path, target_size=(299, 299))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.inception_resnet_v2.preprocess_input(image_array)

    mask = load_img(mask_path, target_size=(299, 299), color_mode='grayscale')
    mask_array = img_to_array(mask)
    mask_array = np.expand_dims(mask_array, axis=0)
    mask_array = tf.keras.applications.inception_resnet_v2.preprocess_input(mask_array)

    # Inpainting: generate the inpainted image
    inpainted_image = inpainting_model.predict([image_array, mask_array])

    # Clip values to be in the valid image range
    inpainted_image = np.clip(inpainted_image, 0, 255).astype(np.uint8)

    return inpainted_image[0]

# Load an example image and mask for inpainting
image_path = 'path/to/image.jpg'
mask_path = 'path/to/mask.png'
inpainted_result = perform_inpainting(image_path, mask_path)

# Display the original image, mask, and inpainted result
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(load_img(image_path))
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(load_img(mask_path, color_mode='grayscale'))
plt.title('Mask')

plt.subplot(1, 3, 3)
plt.imshow(inpainted_result)
plt.title('Inpainted Result')

plt.show()
```

In this script, an InceptionResNetV2 model pre-trained on ImageNet is used for image inpainting. The model takes both the original image and a mask as inputs and generates an inpainted result.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Image inpainting is useful for tasks such as restoring damaged images or removing unwanted objects from photos.

Certainly! Let's explore another advanced concept: image super-resolution using deep learning.

### Image Super-Resolution with Deep Learning:

Image super-resolution involves increasing the resolution of an image to reveal more details. Deep learning models, particularly Generative Adversarial Networks (GANs), have been successfully used for image super-resolution. In this example, we'll use a pre-trained GAN model for image super-resolution.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image super-resolution using a pre-trained model:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

# Load the pre-trained super-resolution model (ESRGAN)
super_resolution_model = tf.keras.models.load_model('path/to/esrgan_model.h5')

# Function to perform image super-resolution
def perform_super_resolution(image_path):
    # Load and preprocess the low-resolution image
    low_res_image = load_img(image_path, target_size=(64, 64))
    low_res_array = img_to_array(low_res_image)
    low_res_array = np.expand_dims(low_res_array, axis=0)
    low_res_array = preprocess_input(low_res_array)

    # Perform super-resolution
    super_res_image = super_resolution_model.predict(low_res_array)

    # Clip values to be in the valid image range
    super_res_image = np.clip(super_res_image, 0, 255).astype(np.uint8)

    return super_res_image[0]

# Load an example low-resolution image for super-resolution
low_res_image_path = 'path/to/low_res_image.jpg'
super_res_result = perform_super_resolution(low_res_image_path)

# Display the original low-resolution image and super-resolution result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(load_img(low_res_image_path))
plt.title('Low-Resolution Image')

plt.subplot(1, 2, 2)
plt.imshow(super_res_result)
plt.title('Super-Resolution Result')

plt.show()
```

In this script, a pre-trained Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) model is used for image super-resolution. The model takes a low-resolution image as input and generates a high-resolution result.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Image super-resolution is beneficial for enhancing the visual quality of images, especially in applications where high-resolution details are crucial.

Certainly! Let's explore another advanced concept: image-to-image translation using Conditional Generative Adversarial Networks (cGANs).

### Image-to-Image Translation with cGANs:

Conditional Generative Adversarial Networks (cGANs) are a type of GAN that can be trained to map images from one domain to another. In this example, we'll use a pre-trained cGAN model for image-to-image translation.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image-to-image translation using a pre-trained cGAN model:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

# Load the pre-trained cGAN model (Pix2Pix)
cGAN_model = tf.keras.models.load_model('path/to/cgan_model.h5')

# Function to perform image-to-image translation
def perform_image_translation(input_image_path):
    # Load and preprocess the input image
    input_image = load_img(input_image_path, target_size=(256, 256))
    input_array = img_to_array(input_image)
    input_array = np.expand_dims(input_array, axis=0)
    input_array = preprocess_input(input_array)

    # Perform image-to-image translation
    translated_image = cGAN_model.predict(input_array)

    # Clip values to be in the valid image range
    translated_image = np.clip(translated_image, 0, 255).astype(np.uint8)

    return translated_image[0]

# Load an example image for image-to-image translation
input_image_path = 'path/to/input_image.jpg'
translation_result = perform_image_translation(input_image_path)

# Display the original input image and translation result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(load_img(input_image_path))
plt.title('Input Image')

plt.subplot(1, 2, 2)
plt.imshow(translation_result)
plt.title('Translation Result')

plt.show()
```

In this script, a pre-trained Pix2Pix cGAN model is used for image-to-image translation. The model takes an input image and generates a translated result in a different domain.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Feel free to experiment with these examples and integrate them into your projects. Image-to-image translation is useful for various applications, such as turning satellite images into maps, converting black and white photos to color, or transforming sketches into realistic images.

Certainly! Let's explore another advanced concept: facial recognition using deep learning.

### Facial Recognition with Deep Learning:

Facial recognition involves identifying and verifying individuals based on their facial features. Deep learning models, particularly Convolutional Neural Networks (CNNs), have been successful in facial recognition tasks. In this example, we'll use a pre-trained deep learning model for facial recognition.

```bash
pip install tensorflow numpy matplotlib opencv-python
```

Now, create a script for facial recognition using a pre-trained model:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained facial recognition model
facial_recognition_model = load_model('path/to/facial_recognition_model.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform facial recognition on an image
def perform_facial_recognition(image_path):
    # Read the input image
    img = cv2.imread(image_path)
  
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
  
    for (x, y, w, h) in faces:
        # Crop the face region from the image
        face_img = img[y:y+h, x:x+w]
    
        # Resize and preprocess the face image for facial recognition model
        face_img = cv2.resize(face_img, (224, 224))
        face_img = image.img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
    
        # Perform facial recognition
        prediction = facial_recognition_model.predict(face_img)
    
        # Get the predicted class label
        predicted_class = np.argmax(prediction)
    
        # Display the result on the image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f'Class: {predicted_class}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with facial recognition results
    cv2.imshow('Facial Recognition Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Perform facial recognition on an example image
input_image_path = 'path/to/input_image.jpg'
perform_facial_recognition(input_image_path)
```

In this script, a pre-trained facial recognition model is used to recognize faces in an input image. The Haar Cascade classifier is used for face detection, and the recognized faces are labeled with their predicted class labels.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Facial recognition has various applications, including security systems, access control, and user authentication in mobile devices. Ensure compliance with privacy and ethical considerations when implementing facial recognition systems.

Certainly! Let's explore another advanced concept: image segmentation for semantic segmentation using DeepLabV3.

### Semantic Image Segmentation with DeepLabV3:

Semantic segmentation involves classifying each pixel in an image into a specific class. DeepLabV3 is a state-of-the-art deep learning model for semantic image segmentation. In this example, we'll use a pre-trained DeepLabV3 model for semantic segmentation.

```bash
pip install tensorflow numpy matplotlib opencv-python
```

Now, create a script for semantic image segmentation using a pre-trained DeepLabV3 model:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the pre-trained DeepLabV3 model
deeplab_model = tf.keras.applications.DeepLabV3(weights='pascal_voc')

# Load the image for segmentation
input_image_path = 'path/to/input_image.jpg'
input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Resize and preprocess the image for segmentation
input_image = cv2.resize(input_image, (513, 513))
input_image = image.img_to_array(input_image)
input_image = np.expand_dims(input_image, axis=0)
input_image = tf.keras.applications.deeplab_v3.preprocess_input(input_image)

# Perform semantic segmentation
segmentation_mask = deeplab_model.predict(input_image)

# Convert the segmentation mask to a colored overlay on the original image
segmentation_mask = segmentation_mask.argmax(axis=-1)
colored_mask = np.zeros_like(input_image[0])
for label, color in zip(range(21), plt.cm.tab20.colors):
    colored_mask[segmentation_mask == label] = color

# Display the original image and segmentation result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(input_image[0] / 255.0)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(colored_mask)
plt.title('Segmentation Result')

plt.show()
```

In this script, a pre-trained DeepLabV3 model is used for semantic image segmentation. The model segments the image into different classes, and a colored overlay is applied to visualize the segmentation result.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Semantic segmentation is crucial for tasks like autonomous vehicles, medical image analysis, and scene understanding in computer vision applications.

Certainly! Let's explore another advanced concept: object tracking using a combination of deep learning and computer vision techniques.

### Object Tracking with DeepSORT:

Object tracking involves following the movement of an object in a sequence of frames in a video. DeepSORT (Deep Simple Online and Realtime Tracking) is an object tracking algorithm that integrates deep learning for object detection and appearance embedding for tracking.

```bash
pip install tensorflow numpy matplotlib opencv-python
```

Now, create a script for object tracking using DeepSORT:

```python
import cv2
import numpy as np
from sort import Sort

# Load the pre-trained object detection model (for example, YOLOv3)
object_detection_model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
object_detection_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
object_detection_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load the COCO class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f]

# Load the pre-trained DeepSORT model
deepsort = Sort()

# Function to perform object tracking in a video
def perform_object_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        object_detection_model.setInput(blob)
        outs = object_detection_model.forward(['yolo_82', 'yolo_94', 'yolo_106'])

        # Extract bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = list(map(int, detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])))
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Perform object tracking using DeepSORT
        if len(boxes) > 0:
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            class_ids = np.array(class_ids)
            trackers = deepsort.update(boxes, confidences, class_ids, frame)

            # Draw bounding boxes and tracking IDs on the frame
            for tracker in trackers:
                x, y, w, h, track_id = tracker
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(frame, str(track_id), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with object tracking results
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Perform object tracking in an example video
video_path = 'path/to/example_video.mp4'
perform_object_tracking(video_path)
```

In this script, the YOLOv3 object detection model is used for initial object detection, and the DeepSORT algorithm is applied for online and real-time object tracking.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Object tracking is essential in various applications, including surveillance, autonomous vehicles, and video analysis. The combination of deep learning for detection and traditional tracking algorithms like DeepSORT can provide robust tracking capabilities.

Certainly! Let's explore another advanced concept: image generation using Generative Adversarial Networks (GANs) for creating realistic synthetic images.

### Image Generation with GANs:

Generative Adversarial Networks (GANs) are powerful models capable of generating new data that resembles a given dataset. In this example, we'll use a pre-trained GAN model to generate synthetic images.

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image generation using a pre-trained GAN model:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained GAN model (for example, StyleGAN2)
gan_model = tf.keras.models.load_model('path/to/stylegan2_model.h5')

# Function to generate synthetic images with the GAN model
def generate_images(num_images=5):
    # Generate random latent vectors as input to the GAN
    latent_vectors = np.random.randn(num_images, gan_model.input_shape[1])

    # Generate synthetic images
    generated_images = gan_model.predict(latent_vectors)

    # Rescale pixel values to the range [0, 255]
    generated_images = (generated_images + 1) * 127.5

    return generated_images.astype(np.uint8)

# Generate synthetic images
num_images_to_generate = 5
generated_images = generate_images(num_images_to_generate)

# Display the generated images
plt.figure(figsize=(15, 3))
for i in range(num_images_to_generate):
    plt.subplot(1, num_images_to_generate, i+1)
    plt.imshow(generated_images[i])
    plt.axis('off')
plt.show()
```

In this script, a pre-trained GAN model (such as StyleGAN2) is used to generate synthetic images. The GAN model takes random latent vectors as input and produces realistic images.

Make sure to adjust the paths and parameters in the script according to your specific use case.

GANs are widely used for image synthesis tasks, including generating realistic faces, objects, or even entire scenes. They have applications in art generation, data augmentation, and more. Note that training GANs requires a substantial amount of data and computational resources, but using pre-trained models allows you to leverage their capabilities without training from scratch.

Certainly! Let's explore another advanced concept: image anomaly detection using autoencoders.

### Image Anomaly Detection with Autoencoders:

Autoencoders are neural network models used for unsupervised learning and dimensionality reduction. In the context of image anomaly detection, an autoencoder is trained on normal images, and anomalies are detected when the model fails to reconstruct an input image accurately. Here's an example:

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image anomaly detection using an autoencoder:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define and train the autoencoder on normal images
def train_autoencoder(normal_images, epochs=20):
    autoencoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(normal_images, normal_images, epochs=epochs, batch_size=32, shuffle=True, validation_split=0.1)

    return autoencoder

# Function to detect anomalies using the trained autoencoder
def detect_anomalies(autoencoder, test_images, threshold=0.02):
    reconstructed_images = autoencoder.predict(test_images)
    mse = np.mean(np.square(test_images - reconstructed_images), axis=(1, 2, 3))
    anomalies = test_images[mse > threshold]

    return anomalies

# Generate synthetic normal images for training
normal_images = np.random.randn(100, 64, 64, 3)
normal_images = (normal_images + 1) * 127.5
normal_images = normal_images.astype(np.uint8)

# Train the autoencoder on normal images
autoencoder = train_autoencoder(normal_images)

# Generate synthetic test images with anomalies
test_images = np.concatenate([np.random.randn(10, 64, 64, 3) * 50 + 128, np.random.randn(10, 64, 64, 3) * 50], axis=0)
test_images = (test_images + 1) * 127.5
test_images = test_images.astype(np.uint8)

# Detect anomalies using the trained autoencoder
anomalies = detect_anomalies(autoencoder, test_images)

# Display the normal and anomalous images
plt.figure(figsize=(15, 3))
for i in range(len(test_images)):
    plt.subplot(2, len(test_images), i+1)
    plt.imshow(test_images[i])
    plt.axis('off')

    if i < len(anomalies):
        plt.subplot(2, len(test_images), len(test_images) + i + 1)
        plt.imshow(anomalies[i])
        plt.axis('off')

plt.show()
```

In this script, the autoencoder is trained on a set of normal images, and anomalies are introduced in the test images. The autoencoder is then used to reconstruct the images, and anomalies are detected based on the reconstruction error.

Make sure to adjust the architecture, parameters, and dataset according to your specific use case.

Anomaly detection using autoencoders is beneficial in various domains, including quality control in manufacturing, intrusion detection in cybersecurity, and anomaly detection in medical imaging. The model learns to reconstruct normal patterns and can identify deviations from the learned normal behavior.

Certainly! Let's explore another advanced concept: image captioning using a combination of computer vision and natural language processing.

### Image Captioning with Neural Image Caption (NIC) Model:

Image captioning involves generating a textual description of the content within an image. A Neural Image Caption (NIC) model combines Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) for language modeling. Here's an example:

```bash
pip install tensorflow numpy matplotlib
```

Now, create a script for image captioning using a pre-trained NIC model:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the pre-trained InceptionV3 model for image feature extraction
image_feature_extraction_model = InceptionV3(weights='imagenet', include_top=False)

# Load the NIC model for image captioning
nic_model = load_model('path/to/nic_model.h5')

# Load the tokenizer for text processing
tokenizer = Tokenizer()
tokenizer.word_index = np.load('path/to/tokenizer_word_index.npy', allow_pickle=True).item()

# Function to preprocess an image for captioning
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to generate a caption for an image
def generate_caption(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Extract image features using InceptionV3
    image_features = image_feature_extraction_model.predict(img_array)
    image_features = tf.reshape(image_features, (image_features.shape[0], -1, image_features.shape[3]))

    # Initialize the caption with the start token
    caption = ['<start>']

    # Generate words until the maximum length or the end token is encountered
    while len(caption) < 20:
        # Tokenize and pad the caption sequence
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=20)

        # Predict the next word in the sequence
        predictions = nic_model.predict([image_features, sequence])
        predicted_word_index = np.argmax(predictions)
    
        # Convert the predicted word index to the corresponding word
        predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_word_index][0]

        # Break if the end token is predicted
        if predicted_word == '<end>':
            break

        # Append the predicted word to the caption
        caption.append(predicted_word)

    # Remove the start and end tokens and join the words to form the final caption
    final_caption = ' '.join(caption[1:-1])

    return final_caption

# Test the image captioning function
example_image_path = 'path/to/example_image.jpg'
predicted_caption = generate_caption(example_image_path)

# Display the original image and predicted caption
img = image.load_img(example_image_path)
plt.imshow(img)
plt.title('Predicted Caption: ' + predicted_caption)
plt.axis('off')
plt.show()
```

In this script, the InceptionV3 model is used for image feature extraction, and an NIC model is employed for caption generation. The tokenizer is used to convert words to indices and vice versa. The `generate_caption` function takes an image path, processes the image, and generates a textual caption.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Image captioning is valuable in applications like accessibility for the visually impaired, content indexing, and enhancing user experiences in image-related tasks. The NIC model combines vision and language to describe the content of images.

Certainly! Let's explore another advanced concept: image-to-text synthesis using OpenAI's CLIP (Contrastive Language–Image Pretraining) model.

### Image-to-Text Synthesis with CLIP:

OpenAI's CLIP model is designed to understand images and text in a unified manner. It can be used for various tasks, including generating textual descriptions for images. Here's an example using the `openai/clip-vit-base-patch16` model from the Hugging Face Transformers library:

```bash
pip install torch torchvision transformers
```

Now, create a script for image-to-text synthesis using the CLIP model:

```python
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# Function to generate textual descriptions for images
def generate_text_from_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    preprocess = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    inputs = processor(images=preprocess(image), return_tensors="pt")

    # Forward pass through the CLIP model
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    label = processor.decode_batch_predictions(outputs.logits_per_image)[0]

    return label

# Test the image-to-text synthesis function
example_image_path = 'path/to/example_image.jpg'
generated_text = generate_text_from_image(example_image_path)

# Display the original image and generated text
img = Image.open(example_image_path)
plt.imshow(img)
plt.title('Generated Text: ' + generated_text)
plt.axis('off')
plt.show()
```

In this script, the CLIP model is used to generate textual descriptions for images. The image is preprocessed using standard transformations, and the CLIP model provides a similarity score and label probabilities. The `generate_text_from_image` function decodes the predicted label to generate a textual description.

Make sure to adjust the paths and parameters in the script according to your specific use case.

CLIP is a versatile model that can be fine-tuned for specific tasks or used out of the box for various vision-language tasks, including image-to-text synthesis. It excels in understanding the relationships between images and text in a contrastive learning framework.

Certainly! Let's explore another advanced concept: text-to-image synthesis using generative models. One popular approach is to use a pre-trained model like BigGAN to generate realistic images from textual descriptions.

### Text-to-Image Synthesis with BigGAN:

```bash
pip install torch torchvision
```

Now, create a script for text-to-image synthesis using BigGAN:

```python
import torch
from torchvision import transforms
from transformers import BigGANModel, BigGANConfig
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained BigGAN model
model = BigGANModel.from_pretrained("google/biggan-deep-256")

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate images from textual descriptions
def generate_image_from_text(text):
    # Tokenize and encode the text
    inputs = model.text_to_image(text, return_tensors="pt").to(device)

    # Generate images from the text
    outputs = model.generate(**inputs)

    # Post-process and display the generated image
    image_array = transforms.ToPILImage()(outputs[0].cpu()).convert("RGB")
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()

# Test the text-to-image synthesis function
text_description = "A colorful bird in a forest"
generate_image_from_text(text_description)
```

In this script, the BigGAN model is used to generate images from textual descriptions. The `generate_image_from_text` function takes a text description, tokenizes it, and generates an image using the pre-trained BigGAN model. The resulting image is then displayed using Matplotlib.

Make sure to adjust the parameters in the script according to your specific use case.

Text-to-image synthesis is a challenging and fascinating task, and BigGAN is one of the models that has shown impressive capabilities in generating high-quality images from textual prompts. This can be applied in creative tasks, such as generating artwork based on textual descriptions.

Certainly! Let's explore another advanced concept: style transfer using neural networks. Style transfer involves applying the artistic style of one image to the content of another image. We can use a pre-trained model like VGG19 to achieve this.

### Image Style Transfer with VGG19:

```bash
pip install torch torchvision matplotlib
```

Now, create a script for style transfer using VGG19:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the content and style images
content_image_path = 'path/to/content_image.jpg'
style_image_path = 'path/to/style_image.jpg'

# Load content and style images
content_image = Image.open(content_image_path)
style_image = Image.open(style_image_path)

# Display the original content and style images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(content_image)
plt.title('Content Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(style_image)
plt.title('Style Image')
plt.axis('off')

plt.show()

# Define the image transformations and normalization
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
])

# Preprocess the content and style images
content_image = preprocess(content_image)
style_image = preprocess(style_image)

# Display the preprocessed content and style images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(content_image.permute(1, 2, 0))
plt.title('Content Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(style_image.permute(1, 2, 0))
plt.title('Style Image')
plt.axis('off')

plt.show()

# Create a VGG19 model with pre-trained weights
vgg19 = models.vgg19(pretrained=True).features.eval()

# Define the content and style layers
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Function to extract features from the specified layers
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if str(name) in layers:
            features[str(name)] = x
    return features

# Define the Gram Matrix computation
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        return gram_matrix

# Define the model for style transfer
class StyleTransferModel(nn.Module):
    def __init__(self, vgg_model, content_layers, style_layers):
        super(StyleTransferModel, self).__init__()
        self.vgg_model = vgg_model
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = 1
        self.style_weight = 1e6

        self.content_losses = {}
        self.style_losses = {}

        self.model = nn.Sequential()
        gram = GramMatrix()

        # Register content and style losses
        i = 0
        for layer in vgg_model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError('Unrecognized layer')

            self.model.add_module(name, layer)

            if name in content_layers:
                target = self.model(content_image).detach()
                content_loss = nn.MSELoss()
                self.model.add_module(f'content_loss_{i}', content_loss)
                self.content_losses[name] = content_loss

            if name in style_layers:
                target_feature = self.model(style_image).detach()
                self.model.add_module(f'gram_{i}', gram)
                style_loss = nn.MSELoss()
                self.model.add_module(f'style_loss_{i}', style_loss)
                self.style_losses[name] = style_loss

    def forward(self, x):
        self.model(x)
        return x

# Function to perform style transfer
def perform_style_transfer(content_image, style_image, num_steps=300, content_weight=1, style_weight=1e6):
    content_image = content_image.unsqueeze(0).to(device)
    style_image = style_image.unsqueeze(0).to(device)

    input_image = content_image.clone().detach().requires_grad_(True).to(device)

    optimizer = optim.LBFGS([input_image])

    model = StyleTransferModel(vgg19, content_layers, style_layers)
    model.to(device)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()
            model(input_image)

            content_loss = 0
            style_loss = 0

            for name, loss in model.content_losses.items():
                content_loss += loss(model.model, content_image)

            for name, loss in model.style_losses.items():
                style_loss += loss(model.model, style_image)

            content_loss *= content_weight
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step [{run[0]}], Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")

            return total_loss

        optimizer.step(closure)

    input_image.data.clamp_(0, 1)
    return input_image.squeeze(0)

# Perform style transfer
stylized_image = perform_style_transfer(content_image, style_image, num_steps=300, content_weight=1, style_weight=1e6)

# Display the original content, style, and stylized images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt

.imshow(content_image.permute(1, 2, 0))
plt.title('Content Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(style_image.permute(1, 2, 0))
plt.title('Style Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(stylized_image.cpu().detach().permute(1, 2, 0))
plt.title('Stylized Image')
plt.axis('off')

plt.show()
```

In this script, VGG19 is used for feature extraction, and a custom model is defined for style transfer. The `perform_style_transfer` function optimizes the input image to minimize the content loss with respect to the content image and the style loss with respect to the style image.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Style transfer allows you to combine the content of one image with the artistic style of another, creating visually appealing and artistic results.

Certainly! Let's continue exploring another advanced concept: image super-resolution using deep learning. Image super-resolution involves increasing the resolution and quality of an image beyond its original size. One popular architecture for this task is the SRGAN (Super-Resolution Generative Adversarial Network). Let's implement a simple example using a pre-trained SRGAN model.

### Image Super-Resolution with SRGAN:

```bash
pip install torch torchvision
```

Now, create a script for image super-resolution using SRGAN:

```python
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vgg19
from torchvision.models.vgg import model_urls
from torch.utils.model_zoo import load_url

# Define the SRGAN model architecture
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define the layers of the generator
        self.layer1 = torch.nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.layer2 = torch.nn.PReLU()
        self.layer3 = self.make_layer(64)
        self.layer4 = self.make_layer(64)
        self.layer5 = torch.nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def make_layer(self, num_channels):
        layers = []
        for _ in range(16):
            layers.append(torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
            layers.append(torch.nn.BatchNorm2d(num_channels, 0.8))
            layers.append(torch.nn.PReLU())
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out5

# Function to load a pre-trained SRGAN model
def load_srgan_model():
    model = Generator()
    model_url = 'https://github.com/VainF/SpaceSR/raw/master/srresnet.pth'
    model.load_state_dict(load_url(model_url))
    return model

# Function to perform super-resolution using SRGAN
def super_resolve(image_path, model):
    image = Image.open(image_path).convert('RGB')

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Preprocess the image
    image = transform(image)
    image = Variable(image.unsqueeze(0))

    # Set the model to evaluation mode
    model.eval()

    # Perform super-resolution
    with torch.no_grad():
        sr_image = model(image)

    # Display the original and super-resolved images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(0).permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sr_image.squeeze(0).permute(1, 2, 0).clamp(0, 1))
    plt.title('Super-Resolved Image')
    plt.axis('off')

    plt.show()

# Load the pre-trained SRGAN model
srgan_model = load_srgan_model()

# Test the super-resolution function
image_path = 'path/to/low_resolution_image.jpg'
super_resolve(image_path, srgan_model)
```

In this script, we define a simple generator network for super-resolution based on the SRGAN architecture. We then load a pre-trained SRGAN model and use it to perform super-resolution on a given low-resolution image.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Image super-resolution is widely used for enhancing the quality and details of images, especially in applications like enhancing surveillance footage or improving the quality of medical images.

Certainly! Let's delve into another advanced concept: facial recognition using deep learning. Facial recognition involves identifying and verifying individuals based on facial features. We'll use a pre-trained face recognition model from the Dlib library.

### Facial Recognition with Dlib:

```bash
pip install dlib
```

Now, create a script for facial recognition using Dlib:

```python
import dlib
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained face recognition model from Dlib
face_recognition_model = dlib.face_recognition_model_v1('shape_predictor_68_face_landmarks.dat')

# Load the pre-trained face detection model from Dlib
face_detector = dlib.get_frontal_face_detector()

# Load an example image for facial recognition
example_image_path = 'path/to/example_image.jpg'
image = cv2.imread(example_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces in the image
faces = face_detector(image_rgb)

# Perform facial recognition for each detected face
for face in faces:
    landmarks = face_recognition_model(image_rgb, face)
  
    # Display the detected face with landmarks
    plt.imshow(image_rgb)
    plt.title('Detected Face')
  
    # Plot facial landmarks
    for landmark in landmarks.parts():
        plt.scatter(landmark.x, landmark.y, s=2, c='red', marker='o')
  
    plt.axis('off')
    plt.show()
```

In this script, we use the Dlib library to perform facial recognition on an example image. The `face_detector` is used to locate faces in the image, and the `face_recognition_model` is used to extract facial landmarks.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Facial recognition has various applications, including identity verification, access control, and surveillance. Dlib provides an efficient implementation for facial recognition with pre-trained models.

Certainly! Let's continue with another advanced concept: object detection using a pre-trained model. We'll use the YOLO (You Only Look Once) model, which is known for its real-time object detection capabilities.

### Object Detection with YOLO (You Only Look Once):

```bash
pip install opencv-python
```

Now, create a script for object detection using YOLO:

```python
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained YOLO model and configuration
yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
yolo_classes = []
with open('coco.names', 'r') as f:
    yolo_classes = [line.strip() for line in f]

# Load an example image for object detection
example_image_path = 'path/to/example_image.jpg'
image = cv2.imread(example_image_path)
height, width = image.shape[:2]

# Set input blob for the YOLO model
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo_net.setInput(blob)

# Get output layer names
out_layer_names = yolo_net.getUnconnectedOutLayersNames()

# Forward pass and get predictions
detections = yolo_net.forward(out_layer_names)

# Filter and process detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(obj[0] * width)
            center_y = int(obj[1] * height)
            w = int(obj[2] * width)
            h = int(obj[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw rectangle and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{yolo_classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with object detection results
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Object Detection with YOLO')
plt.axis('off')
plt.show()
```

In this script, we use the OpenCV library to perform object detection with the YOLO model on an example image. The YOLO model is pre-trained on the COCO dataset, and we load the model weights, configuration, and class names.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Object detection is crucial in computer vision applications, including surveillance, autonomous vehicles, and image understanding. YOLO is a popular choice for real-time object detection due to its speed and accuracy.

Certainly! Let's explore another advanced concept: image segmentation using a pre-trained DeepLabv3 model. Image segmentation involves partitioning an image into meaningful segments or regions. DeepLabv3 is a state-of-the-art model for semantic segmentation.

### Image Segmentation with DeepLabv3:

```bash
pip install tensorflow matplotlib
```

Now, create a script for image segmentation using DeepLabv3:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained DeepLabv3 model
model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(None, None, 3))

# Load an example image for image segmentation
example_image_path = 'path/to/example_image.jpg'
image = Image.open(example_image_path)
image_array = np.array(image)

# Preprocess the image for model input
input_image = tf.image.resize(image_array, (256, 256))
input_image = tf.expand_dims(input_image, 0)
input_image = tf.keras.applications.densenet.preprocess_input(input_image)

# Perform image segmentation
predictions = model.predict(input_image)

# Get the segmentation mask
mask = tf.argmax(predictions, axis=-1)
mask = mask[0, :, :, np.newaxis]

# Display the original image and segmentation mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='viridis', alpha=0.7)
plt.title('Segmentation Mask')
plt.axis('off')

plt.show()
```

In this script, we use a pre-trained DenseNet201 model from TensorFlow's Keras applications for image segmentation. The example image is preprocessed and fed into the model, and the resulting segmentation mask is displayed alongside the original image.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Image segmentation is widely used in applications like object recognition, medical imaging, and scene understanding. DeepLabv3, with its deep convolutional architecture, provides accurate and detailed segmentation results.

Certainly! Let's continue exploring another advanced concept: instance segmentation using the Mask R-CNN (Mask Region-based Convolutional Neural Network) model. Instance segmentation goes a step further than semantic segmentation by not only segmenting objects but also distinguishing between individual instances of the same object.

### Instance Segmentation with Mask R-CNN:

```bash
pip install tensorflow matplotlib
```

Now, create a script for instance segmentation using Mask R-CNN:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mrcnn import model as modellib, utils
from mrcnn.config import Config

# Load the pre-trained Mask R-CNN model
class InferenceConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True)

# Load an example image for instance segmentation
example_image_path = 'path/to/example_image.jpg'
image = Image.open(example_image_path)
image_array = np.array(image)

# Perform instance segmentation
results = model.detect([image_array], verbose=0)
r = results[0]

# Display the original image with instance segmentation results
plt.imshow(image_array)
plt.title('Instance Segmentation Results')
plt.axis('off')

# Plot bounding boxes, masks, and class labels
for i in range(r['rois'].shape[0]):
    mask = r['masks'][:, :, i]
    plt.contour(mask, colors='r', linewidths=1, alpha=0.7)
  
    y1, x1, y2, x2 = r['rois'][i]
    plt.annotate(
        f"{config.CLASS_NAMES[r['class_ids'][i]]}",
        (x1, y1),
        color='white', weight='bold',
        fontsize=8, ha='left', va='top',
        bbox=dict(boxstyle="round", alpha=0.7, pad=0, edgecolor='none', facecolor='red')
    )

plt.show()
```

In this script, we use the Mask R-CNN model with pre-trained weights for instance segmentation. The example image is processed, and the model detects and segments instances of objects, including bounding boxes, masks, and class labels.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Instance segmentation is valuable in applications such as object counting, tracking, and detailed understanding of scenes. Mask R-CNN is a powerful model for achieving accurate instance segmentation results.

Certainly! Let's explore another advanced concept: anomaly detection in images using autoencoders. An autoencoder is a type of neural network that learns to encode input data into a lower-dimensional representation and then decode it back to the original data. Anomalies can be detected when the reconstruction error is high.

### Anomaly Detection with Autoencoders:

```bash
pip install tensorflow matplotlib
```

Now, create a script for anomaly detection using autoencoders:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load an example dataset (normal images for training)
normal_images_path = 'path/to/normal_images/'
normal_images = [np.array(Image.open(normal_images_path + f)) for f in os.listdir(normal_images_path)]

# Normalize pixel values to be between 0 and 1
normal_images = np.array(normal_images) / 255.0

# Split data into training and validation sets
train_size = int(0.8 * len(normal_images))
train_normal_images, val_normal_images = normal_images[:train_size], normal_images[train_size:]

# Define and train the autoencoder model
input_shape = (256, 256, 3)  # Adjust according to your image dimensions
autoencoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(train_normal_images, train_normal_images, epochs=10, batch_size=32, validation_data=(val_normal_images, val_normal_images))

# Evaluate the autoencoder on normal and anomalous images
anomalous_image_path = 'path/to/anomalous_image.jpg'
anomalous_image = np.array([np.array(Image.open(anomalous_image_path))]) / 255.0

# Get predictions for normal and anomalous images
normal_predictions = autoencoder.predict(val_normal_images)
anomalous_prediction = autoencoder.predict(anomalous_image)

# Calculate reconstruction errors
normal_errors = np.mean(np.square(val_normal_images - normal_predictions))
anomalous_error = np.mean(np.square(anomalous_image - anomalous_prediction))

# Set a threshold for anomaly detection
threshold = 0.02

# Detect anomalies
is_anomalous = anomalous_error > threshold

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(val_normal_images[0])
plt.title('Normal Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(anomalous_image[0])
plt.title('Anomalous Image\nAnomaly Detected: {}'.format(is_anomalous))
plt.axis('off')

plt.show()
```

In this script, we train an autoencoder on a dataset of normal images and then use it to reconstruct both normal and anomalous images. The reconstruction error is calculated, and an anomaly is detected if the error is above a specified threshold.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Anomaly detection with autoencoders is useful for identifying unusual patterns or outliers in image data, making it applicable in areas such as quality control and anomaly monitoring.

Certainly! Let's explore another advanced concept: neural style transfer. Neural style transfer is a technique that uses convolutional neural networks to apply the artistic style of one image to the content of another image.

### Neural Style Transfer:

```bash
pip install tensorflow matplotlib
```

Now, create a script for neural style transfer:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load content and style images
content_image_path = 'path/to/content_image.jpg'
style_image_path = 'path/to/style_image.jpg'

content_image = Image.open(content_image_path)
style_image = Image.open(style_image_path)

# Display the original content and style images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(content_image)
plt.title('Content Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(style_image)
plt.title('Style Image')
plt.axis('off')

plt.show()

# Convert images to arrays and normalize pixel values
content_array = np.array(content_image) / 255.0
style_array = np.array(style_image) / 255.0

# Add batch dimension to the images
content_array = np.expand_dims(content_array, axis=0)
style_array = np.expand_dims(style_array, axis=0)

# Define the VGG19 model for feature extraction
vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Extract features from the content and style images
content_features = vgg19(content_array)
style_features = vgg19(style_array)

# Define a function to compute the Gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)

# Define the style transfer model
class StyleTransferModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleTransferModel, self).__init__()
        self.vgg = vgg19
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        return {'content': content_outputs, 'style': style_outputs}

# Define the content and style layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Create the style transfer model
style_transfer_model = StyleTransferModel(style_layers, content_layers)

# Define the optimizer and loss functions
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= 1e-2
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= 1e-1
    loss = style_loss + content_loss
    return loss

# Define the training step
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = style_transfer_model(image)
        loss = style_content_loss(outputs)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

# Convert content and style target features to dictionaries
style_targets = style_transfer_model(style_array)['style']
content_targets = style_transfer_model(content_array)['content']

# Initialize the generated image with the content image
generated_image = tf.Variable(content_array)

# Specify the number of training iterations
num_iterations = 1000

# Perform neural style transfer
for _ in range(num_iterations):
    train_step(generated_image)

# Display the generated image
plt.imshow(generated_image.numpy()[0])
plt.title('Generated Image')
plt.axis('off')
plt.show()
```

In this script, we perform neural style transfer by optimizing an image to minimize the difference between its features and the features of the content image while matching the style of the style image.

Make sure to adjust the paths and parameters in the script according to your specific use case.

Neural style transfer is a creative technique that allows you to apply artistic styles to images, producing visually appealing and unique results.

Certainly! Let's explore another advanced concept: Generative Adversarial Networks (GANs). GANs consist of two neural networks, a generator, and a discriminator, that are trained simultaneously through adversarial training to generate realistic-looking data. We'll create a simple GAN for generating images.

### Generative Adversarial Network (GAN) for Image Generation:

```bash
pip install tensorflow matplotlib
```

Now, create a script for a simple GAN:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Define the generator model
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Define the discriminator model
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Define the combined GAN model
latent_dim = 100
img_shape = (28, 28, 1)

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Load the MNIST dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=3)

# Training parameters
batch_size = 64
half_batch = batch_size // 2
epochs = 30000

# Training loop
for epoch in range(epochs):

    # Train the discriminator
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    imgs = X_train[idx]

    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    gen_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.array([1] * batch_size)

    g_loss = gan.train_on_batch(noise, valid_y)

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

    # Save generated images at specified intervals
    if epoch % 1000 == 0:
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
```

In this script, we define a simple GAN architecture for generating images resembling the MNIST dataset. The generator creates fake images, and the discriminator distinguishes between real and fake images. The GAN is trained iteratively to improve the quality of the

generated images.

Make sure to adjust the model architectures, training parameters, and dataset according to your specific use case.

GANs are widely used for generating realistic-looking data, such as images, and have applications in various fields, including art generation, data augmentation, and image synthesis.

Certainly! Let's delve into another exciting topic: Transfer Learning. Transfer learning involves using a pre-trained model on a large dataset and fine-tuning it for a specific task or domain. We'll use a pre-trained Convolutional Neural Network (CNN) for image classification and fine-tune it on a different dataset.

### Transfer Learning for Image Classification:

```bash
pip install tensorflow matplotlib
```

Now, create a script for transfer learning:

```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base
base_model.trainable = False

# Create a custom model with the pre-trained base and additional layers
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Load your dataset
# Assuming you have a directory structure like:
# - dataset/
#   - train/
#     - cats/
#     - dogs/
#   - validation/
#     - cats/
#     - dogs/
train_dir = 'path/to/dataset/train'
validation_dir = 'path/to/dataset/validation'

# Use ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Specify the batch size
batch_size = 32

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

# Specify the number of training and validation steps
train_steps = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Train the model with fine-tuning
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Plot the training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

In this script, we use the VGG16 model pre-trained on ImageNet as the convolutional base. We freeze the weights of the base model and add custom layers for classification. The model is then fine-tuned on a different dataset using data augmentation.

Make sure to adjust the paths, dataset structure, and parameters in the script according to your specific use case.

Transfer learning is a powerful technique when you have limited data for a specific task. By leveraging pre-trained models, you can benefit from the knowledge learned on large datasets and adapt it to your own problem.





Certainly! Let's explore another concept: Natural Language Processing (NLP) with deep learning. We'll use a pre-trained language model for text classification.

### Natural Language Processing (NLP) with BERT:

```bash
pip install transformers tensorflow matplotlib
```

Now, create a script for text classification using BERT:

```python
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load and preprocess your text data
# For this example, let's assume you have a DataFrame 'df' with 'text' and 'label' columns
# df = pd.read_csv('path/to/your/dataset.csv')

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize and encode the text data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='tf')
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='tf')

# Convert labels to TensorFlow format
train_labels = tf.convert_to_tensor(train_labels.tolist())
test_labels = tf.convert_to_tensor(test_labels.tolist())

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train the model
history = model.fit(train_encodings, train_labels, epochs=3, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_encodings, test_labels)
print(f"Test Accuracy: {test_acc}")

# Plot the training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions on new text data
new_texts = ['Your new text data goes here.', 'Another example text.']
new_encodings = tokenizer(new_texts, truncation=True, padding=True, max_length=128, return_tensors='tf')
predictions = model.predict(new_encodings)
predicted_labels = tf.argmax(predictions.logits, axis=1)

# Print predicted labels
print("Predicted Labels:", predicted_labels.numpy())
```

In this script, we use the Hugging Face `transformers` library to load a pre-trained BERT model for sequence classification. The script demonstrates how to tokenize and encode text data, compile and train the model, evaluate its performance, and make predictions on new text data.

Make sure to replace the dataset loading part with your actual dataset and adjust the parameters based on your specific use case.

BERT (Bidirectional Encoder Representations from Transformers) is a powerful pre-trained language model that has shown excellent performance on various NLP tasks. Fine-tuning BERT for text classification allows you to leverage its contextual understanding for specific domains or applications.




Certainly! Let's explore another advanced NLP concept: Named Entity Recognition (NER) using spaCy. Named Entity Recognition is the task of identifying entities such as names, locations, organizations, and more in text.

### Named Entity Recognition (NER) with spaCy:

```bash
pip install spacy matplotlib
python -m spacy download en_core_web_sm
```

Now, create a script for Named Entity Recognition using spaCy:

```python
import spacy
from spacy import displacy
import matplotlib.pyplot as plt

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Process text data
text = "Apple Inc. is planning to open a new store in New York City."
doc = nlp(text)

# Visualize named entities
displacy.render(doc, style='ent', jupyter=False)
plt.show()

# Extract named entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Named Entities:", entities)
```

In this script, we use spaCy to perform Named Entity Recognition on a sample text. The `en_core_web_sm` model is a small English language model provided by spaCy.

Make sure to adjust the text variable with your own text data for NER.

Named Entity Recognition is crucial for extracting structured information from unstructured text, and spaCy provides an efficient and accurate solution for this task.

Feel free to incorporate this into your NLP pipelines or applications where entity recognition is essential.




Certainly! Let's explore another advanced NLP concept: Sentiment Analysis. Sentiment Analysis involves determining the sentiment expressed in a piece of text, whether it's positive, negative, or neutral.

### Sentiment Analysis with VADER (Valence Aware Dictionary and sEntiment Reasoner):

```bash
pip install vaderSentiment matplotlib
```

Now, create a script for Sentiment Analysis using VADER:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Instantiate the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment for a sample text
text = "I love using this product! It's fantastic."
compound_score = analyzer.polarity_scores(text)['compound']

# Display sentiment analysis results
if compound_score >= 0.05:
    sentiment = 'Positive'
elif compound_score <= -0.05:
    sentiment = 'Negative'
else:
    sentiment = 'Neutral'

print(f"Sentiment: {sentiment}")
print("Sentiment Scores:", analyzer.polarity_scores(text))

# Visualize sentiment scores
labels = ['Positive', 'Neutral', 'Negative']
scores = [analyzer.polarity_scores(text)['pos'], analyzer.polarity_scores(text)['neu'], analyzer.polarity_scores(text)['neg']]

plt.bar(labels, scores, color=['green', 'gray', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Score')
plt.title('Sentiment Analysis')
plt.show()
```

In this script, we use the VADER sentiment analysis tool, which is a lexicon and rule-based sentiment analysis tool designed for social media text. The `SentimentIntensityAnalyzer` assigns a compound score that represents the overall sentiment of the text.

Make sure to adjust the text variable with your own text data for sentiment analysis.

VADER is particularly useful for quick sentiment analysis tasks, especially in scenarios where you need to analyze text with informal language, such as social media content.

Feel free to incorporate this into your applications or workflows where sentiment analysis is required.





Certainly! Let's explore another advanced NLP concept: Text Summarization. Text Summarization involves condensing a longer document into a shorter version while retaining its essential information.

### Text Summarization with Gensim's TextRank Algorithm:

```bash
pip install gensim
```

Now, create a script for text summarization using Gensim's TextRank algorithm:

```python
from gensim.summarization import summarize

# Sample text for summarization
text = """
Gensim is an open-source Python library for vector space modeling and topic modeling. 
It is designed to handle large text collections using data streaming and incremental algorithms.
TextRank is an extractive summarization algorithm implemented in Gensim. It works by analyzing the co-occurrence 
of words in the input text and identifying the most important sentences based on their centrality in the network of relationships.
Gensim's implementation of TextRank is a popular choice for automatic summarization tasks.
"""

# Summarize the text
summary = summarize(text, ratio=0.3)

# Display the original and summarized text
print("Original Text:\n", text)
print("\nSummarized Text:\n", summary)
```

In this script, we use Gensim's TextRank algorithm for extractive summarization. The `summarize` function takes a text input and returns a summary based on the most important sentences.

You can adjust the `ratio` parameter in the `summarize` function to control the length of the summary.

Text summarization is useful for condensing lengthy documents, articles, or reports, making it easier for users to grasp the main points quickly.

Feel free to incorporate this into your applications or workflows where text summarization is required.





Certainly! Let's explore another advanced NLP concept: Machine Translation using the Transformer model. We'll use the popular library `transformers` by Hugging Face to leverage a pre-trained Transformer model for translation.

### Machine Translation with Hugging Face Transformers:

```bash
pip install transformers
```

Now, create a script for machine translation:

```python
from transformers import pipeline

# Load the translation pipeline for English to French
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# Sample English text for translation
english_text = "Machine translation is the task of automatically converting text from one language into another."

# Translate the text to French
french_translation = translator(english_text, max_length=500)[0]['translation_text']

# Display the original and translated text
print("Original Text (English):\n", english_text)
print("\nTranslated Text (French):\n", french_translation)
```

In this script, we use the `transformers` library to load a pre-trained machine translation model (Helsinki-NLP/opus-mt-en-fr) for translating English to French. The `pipeline` function simplifies the process of loading and using the model.

You can explore various language pairs and models provided by Hugging Face for different translation tasks.

Machine translation is essential for breaking down language barriers and enabling communication across different languages.

Feel free to adapt this script for your specific translation needs or integrate it into applications where translation functionality is required.





Certainly! Let's explore another advanced NLP concept: Question Answering with the Transformer-based model, specifically using the Hugging Face Transformers library.

### Question Answering with Hugging Face Transformers:

```bash
pip install transformers
```

Now, create a script for question answering:

```python
from transformers import pipeline

# Load the question answering pipeline for English
question_answerer = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Sample context (passage) for question answering
context = """
The Hugging Face Transformers library is a popular and powerful library for natural language processing (NLP) 
tasks using state-of-the-art transformer models. It provides easy-to-use interfaces for a variety of NLP tasks 
such as text classification, named entity recognition, machine translation, and question answering.
"""

# Ask a question related to the context
question = "What is the Hugging Face Transformers library used for?"

# Get the answer to the question
answer = question_answerer(question=question, context=context)['answer']

# Display the context, question, and answer
print("Context:\n", context)
print("\nQuestion:\n", question)
print("\nAnswer:\n", answer)
```

In this script, we use the `transformers` library to load a pre-trained question-answering model (bert-large-uncased-whole-word-masking-finetuned-squad). The `pipeline` function simplifies the process of loading and using the model.

You can adapt the script to handle different contexts and questions.

Question answering models are valuable for extracting information from a given context, making them suitable for a variety of applications, including chatbots, virtual assistants, and information retrieval systems.

Feel free to integrate this script into your projects where question answering functionality is needed.





Certainly! Let's explore another advanced concept: Speech Recognition using the SpeechRecognition library. Speech recognition involves converting spoken language into written text, and the SpeechRecognition library makes it easy to work with various speech recognition engines.

### Speech Recognition with SpeechRecognition Library:

```bash
pip install SpeechRecognition
```

Now, create a script for speech recognition:

```python
import speech_recognition as sr

# Create a speech recognition object
recognizer = sr.Recognizer()

# Record audio from the microphone
with sr.Microphone() as source:
    print("Say something...")
    audio = recognizer.listen(source, timeout=5)  # Adjust the timeout as needed

# Perform speech recognition
try:
    text = recognizer.recognize_google(audio)
    print("You said:", text)
except sr.UnknownValueError:
    print("Sorry, could not understand audio.")
except sr.RequestError as e:
    print(f"Error connecting to Google API: {e}")
```

In this script, we use the SpeechRecognition library to capture audio from the microphone and perform speech recognition using Google's speech recognition API. You can also use other engines supported by SpeechRecognition, such as Sphinx, Wit.ai, or Microsoft Bing Voice Recognition.

Adjust the `timeout` parameter in the `listen` method based on the expected length of the spoken input.

Speech recognition is valuable for applications like voice-controlled assistants, transcription services, and more.

Feel free to incorporate this script into your projects where speech recognition capabilities are needed.




Certainly! Let's explore another advanced concept: Object Detection using a pre-trained Convolutional Neural Network (CNN) and the OpenCV library. Object detection involves identifying and locating objects within an image.

### Object Detection with OpenCV:

```bash
pip install opencv-python
```

Now, create a script for object detection:

```python
import cv2

# Load the pre-trained YOLO (You Only Look Once) model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Download YOLO weights and configuration file
classes = []
with open("coco.names", "r") as f:  # Download COCO names file
    classes = [line.strip() for line in f.readlines()]

# Load the image
image = cv2.imread("object_detection_example.jpg")  # Replace with your image file

# Get image dimensions
height, width, _ = image.shape

# Convert the image to a blob
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)

# Set the input to the YOLO network
net.setInput(blob)

# Get output layer names
output_layers = net.getUnconnectedOutLayersNames()

# Run forward pass to get output from YOLO
outs = net.forward(output_layers)

# Process the outputs to get bounding boxes, confidences, and class IDs
conf_threshold = 0.5
nms_threshold = 0.4

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes on the image
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box
    label = classes[class_ids[i]]
    confidence = confidences[i]
    color = (0, 255, 0)  # Green color
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the result
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

In this script, we use the YOLO (You Only Look Once) model for object detection. The YOLO model is pre-trained on the COCO dataset and can detect a wide range of objects. Make sure to download the YOLO weights, configuration file, and COCO names file from the official YOLO website (https://pjreddie.com/darknet/yolo/).

Replace the image file ("object_detection_example.jpg") with your own image file.

Object detection is widely used in applications such as surveillance, autonomous vehicles, and image analysis. Feel free to incorporate this script into your projects where object detection capabilities are required.





Certainly! Let's explore another advanced concept: Facial Recognition using a pre-trained model and the face_recognition library. Facial recognition involves identifying and verifying individuals based on their facial features.

### Facial Recognition with face_recognition Library:

```bash
pip install face_recognition
```

Now, create a script for facial recognition:

```python
import face_recognition
import cv2

# Load an image with faces
image_path = "facial_recognition_example.jpg"  # Replace with your image file
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load face encodings for known individuals
known_face_encodings = []
known_face_names = []

# Add known faces (replace these with the faces you want to recognize)
# Example: known_face_encodings.append(face_recognition.face_encodings(known_image)[0])
# Example: known_face_names.append("John Doe")

# Sample known face encoding (commented out for demonstration purposes)
# known_face_encodings.append(...)
# known_face_names.append("John Doe")

# Locate faces in the image
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# Loop through each face found in the image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Check if the face matches any known face
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"  # Default to unknown if no match is found

    # If a match is found, use the name of the known face
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a rectangle around the face and display the name
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# Display the result
cv2.imshow("Facial Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

In this script, we use the face_recognition library, which is built on top of the popular dlib library, to perform facial recognition. Ensure that you have face images of known individuals and their corresponding face encodings to compare with the faces in the input image.

Replace the image file ("facial_recognition_example.jpg") with your own image file and add known face encodings and names as needed.

Facial recognition is used in various applications, including security systems, access control, and identity verification. Feel free to incorporate this script into your projects where facial recognition capabilities are required.




Certainly! Below is a more comprehensive script that includes the full code for facial recognition using the `face_recognition` library:

```python
import face_recognition
import cv2

# Function to load known faces and their encodings
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # Add known faces (replace these with the faces you want to recognize)
    # Example: known_face_encodings.append(face_recognition.face_encodings(known_image)[0])
    # Example: known_face_names.append("John Doe")

    # Sample known face encoding (commented out for demonstration purposes)
    # known_face_encodings.append(...)
    # known_face_names.append("John Doe")

    return known_face_encodings, known_face_names

# Function to perform facial recognition on an image
def perform_facial_recognition(image_path, known_face_encodings, known_face_names):
    # Load the image with faces
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Locate faces in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Loop through each face found in the image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default to unknown if no match is found

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the result
    cv2.imshow("Facial Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform facial recognition on an image
    image_path = "facial_recognition_example.jpg"  # Replace with your image file
    perform_facial_recognition(image_path, known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this script:

- The `load_known_faces` function is used to load known faces and their encodings. You can add known faces by replacing the placeholder code with actual face images.
- The `perform_facial_recognition` function takes an image path and performs facial recognition using the loaded known faces and encodings.
- The `main` function serves as the entry point to the script.

Replace the image file ("facial_recognition_example.jpg") with your own image file. Make sure to add known faces with their respective encodings to the `load_known_faces` function.

Feel free to integrate this script into your projects where facial recognition functionality is needed.




Certainly! Let's extend the script to include real-time facial recognition using a webcam. We'll continuously capture frames from the webcam, perform facial recognition on each frame, and display the results in real-time.

```python
import face_recognition
import cv2

# Function to load known faces and their encodings
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # Add known faces (replace these with the faces you want to recognize)
    # Example: known_face_encodings.append(face_recognition.face_encodings(known_image)[0])
    # Example: known_face_names.append("John Doe")

    # Sample known face encoding (commented out for demonstration purposes)
    # known_face_encodings.append(...)
    # known_face_names.append("John Doe")

    return known_face_encodings, known_face_names

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"  # Default to unknown if no match is found

            # If a match is found, use the name of the known face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this extended script:

- The `perform_realtime_facial_recognition` function continuously captures frames from the webcam, performs facial recognition on each frame, and displays the results in real-time.
- The webcam stream is opened using `cv2.VideoCapture(0)`, where `0` represents the default camera. Adjust this value if you have multiple cameras.

Run this script, and it will display real-time facial recognition results from the webcam. Press 'q' to exit the application.

Make sure to adjust the known faces and their encodings in the `load_known_faces` function based on your requirements.





Certainly! Let's add the ability to dynamically update the known faces during real-time facial recognition. We'll introduce a key to trigger the addition of a new face to the known faces. This will allow you to continuously update the set of known faces without restarting the application.

```python
import face_recognition
import cv2

# Function to load known faces and their encodings
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # Add known faces (replace these with the faces you want to recognize)
    # Example: known_face_encodings.append(face_recognition.face_encodings(known_image)[0])
    # Example: known_face_names.append("John Doe")

    # Sample known face encoding (commented out for demonstration purposes)
    # known_face_encodings.append(...)
    # known_face_names.append("John Doe")

    return known_face_encodings, known_face_names

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"  # Default to unknown if no match is found

            # If a match is found, use the name of the known face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add a new face
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this extended script:

- The `add_new_face` function allows you to dynamically add a new face to the known faces during real-time facial recognition.
- Press 'a' to trigger the addition of a new face. The script will prompt you to enter the name for the new face.
- Press 'q' to exit the application.

This enhancement provides a more interactive experience, allowing you to update the set of known faces without restarting the application.





Certainly! Let's enhance the script to include face recognition confidence levels and display them alongside the recognized names. Additionally, we'll add the option to dynamically remove a face from the known faces during real-time facial recognition.

```python
import face_recognition
import cv2

# Function to load known faces and their encodings
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # Add known faces (replace these with the faces you want to recognize)
    # Example: known_face_encodings.append(face_recognition.face_encodings(known_image)[0])
    # Example: known_face_names.append("John Doe")

    # Sample known face encoding (commented out for demonstration purposes)
    # known_face_encodings.append(...)
    # known_face_names.append("John Doe")

    return known_face_encodings, known_face_names

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add or remove faces
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this enhanced script:

- The `remove_face` function allows you to dynamically remove a face from the known faces during real-time facial recognition.
- Press 'r' to trigger the removal of a face. The script will prompt you to enter the name of the face to remove.
- Face recognition confidence levels are displayed alongside recognized names. The `tolerance` parameter is adjusted to control the matching sensitivity.

This script provides a more interactive experience, allowing you to dynamically update, add, and remove faces during real-time facial recognition.





Certainly! Let's further enhance the script by adding the option to save and load the known faces and their encodings to and from a file. This feature will allow you to persist the known faces across different runs of the application.

```python
import face_recognition
import cv2
import pickle

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add or remove faces
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break



    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this enhanced script:

- The known faces and their encodings are saved to and loaded from a file using the `save_known_faces` and `load_known_faces` functions.
- The `save_known_faces` function is called whenever a new face is added or removed, ensuring that changes persist across runs.
- The `KNOWN_FACES_FILE` constant specifies the file path for storing the known faces data. Adjust this path as needed.

With this enhancement, the known faces will be saved to the file, allowing you to maintain a persistent set of known faces across different executions of the script.




Certainly! Let's continue by adding an option to reset the known faces, clearing all stored face data. This can be useful if you want to start fresh with a new set of known faces.

```python
import face_recognition
import cv2
import pickle
import os

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
         

 # Reset known faces, clearing all stored data
            reset_known_faces()

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The `reset_known_faces` function is introduced to reset known faces, clearing all stored face data.
- Press 'c' to trigger the reset of known faces. This option is useful when you want to start fresh with a new set of known faces.

This enhancement adds the capability to reset the known faces, providing more flexibility when managing the data used for facial recognition.




Certainly! Let's continue by improving the script to handle the case when no face is detected in the input image or webcam frame. We'll add a message to inform the user when no faces are found.

```python
import face_recognition
import cv2
import pickle
import os

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
         

 break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this update:

- The script checks if any faces are found in the frame, and if not, it prints a message and continues to the next frame.
- This enhancement ensures that the user is informed when no faces are detected, preventing unnecessary processing and improving user feedback.

Feel free to run the script and test these improvements!





Certainly! Let's continue by adding a feature to capture and save snapshots when a known face is recognized. This can be useful for creating a log or album of recognized faces.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot when a known face is recognized
                save_snapshot(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        # Break the loop if 'q' is pressed


        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script creates a `SNAPSHOT_DIR` directory to save snapshots. You can change this directory to your preferred location.
- The `save_snapshot` function saves a snapshot of the recognized face with the person's name and a timestamp.
- When a known face is recognized, a snapshot is saved to the specified directory.

Feel free to run the updated script and test the new snapshot feature!





Certainly! Let's continue by enhancing the script to include the ability to display the saved snapshots in a separate window. This will allow you to see the snapshots in real-time as they are captured.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot when a known face is recognized
                save_snapshot(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, display snapshots, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()



# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this update:

- A new function `display_snapshots` is introduced to display saved snapshots in a separate window.
- Press 'd' during real-time facial recognition to display saved snapshots.
- The snapshots will be displayed for 2 seconds each before moving to the next one.

Feel free to run the updated script and explore the new feature!





Certainly! Let's continue by improving the script to include the option to delete individual snapshots or all snapshots in the `SNAPSHOT_DIR`. This feature allows better management of saved snapshots.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_face_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot when a known face is recognized
                save_snapshot(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2

)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, display snapshots, delete snapshots, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces functions `delete_snapshot` and `delete_all_snapshots` to delete individual snapshots or all snapshots, respectively.
- Press 'x' during real-time facial recognition to delete specific snapshots or all snapshots in the `SNAPSHOT_DIR`.
- The user is prompted to enter the snapshot filename to delete or enter 'all' to delete all snapshots.

Feel free to run the updated script and explore the new snapshot management features!




Certainly! Let's continue by adding the ability to record video clips when a known face is recognized. This can be useful for capturing moments when specific individuals are detected.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write(frame)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
         

 text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, display snapshots,
        # delete snapshots, display video clips, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces functions `create_video_directory`, `save_video_clip`, and `display_video_clips` to manage video clips.
- Press 'v' during real-time facial recognition to display saved video clips.
- Video clips are recorded for 2.5 seconds (50 frames) when a known face is recognized.
- Press 'Esc' to exit the video clip display.

Feel free to run the updated script and explore the new video clip management features!





Certainly! Let's continue by enhancing the script to include the option to delete individual video clips or all video clips in the `VIDEO_DIR`. This feature allows better management of saved video clips.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write(frame

)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to delete a specific video clip
def delete_video_clip(video_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    try:
        os.remove(video_path)
        print(f"Video clip deleted: {video_filename}")
    except FileNotFoundError:
        print(f"Video clip not found: {video_filename}")

# Function to delete all video clips in the video directory
def delete_all_video_clips():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        os.remove(video_path)
    print("All video clips deleted.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, display snapshots,
        # delete snapshots, display video clips, delete video clips, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video clip filename to delete
            video_filename = input("Enter the video clip filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces functions `delete_video_clip` and `delete_all_video_clips` to delete individual video clips or all video clips, respectively.
- Press 'y' during real-time facial recognition to delete specific video clips or all video clips in the `VIDEO_DIR`.
- The user is prompted to enter the video clip filename to delete or enter 'all' to delete all video clips.

Feel free to run the updated script and explore the new video clip management





Certainly! Let's enhance the script further by adding the ability to delete individual video clips or all video clips in the `VIDEO_DIR`. This feature will provide better management of saved video clips.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write(frame

)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to delete a specific video clip
def delete_video_clip(video_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    try:
        os.remove(video_path)
        print(f"Video clip deleted: {video_filename}")
    except FileNotFoundError:
        print(f"Video clip not found: {video_filename}")

# Function to delete all video clips in the video directory
def delete_all_video_clips():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        os.remove(video_path)
    print("All video clips deleted.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, display snapshots,
        # delete snapshots, display video clips, delete video clips, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video clip filename to delete
            video_filename = input("Enter the video clip filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces functions `delete_video_clip` and `delete_all_video_clips` to delete individual video clips or all video clips, respectively.
- Press 'y' during real-time facial recognition to delete specific video clips or all video clips in the `VIDEO_DIR`.
- The user is prompted to enter the video clip filename to delete or enter 'all' to delete all video clips.

Feel free to run the updated script and explore the new video clip management




Certainly! Let's continue by incorporating the ability to delete specific video clips or all video clips in the `VIDEO_DIR`. This will provide you with more control over the management of recorded video clips.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write

(frame)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to delete a specific video clip
def delete_video_clip(video_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    try:
        os.remove(video_path)
        print(f"Video clip deleted: {video_filename}")
    except FileNotFoundError:
        print(f"Video clip not found: {video_filename}")

# Function to delete all video clips in the video directory
def delete_all_video_clips():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        os.remove(video_path)
    print("All video clips deleted.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, display snapshots,
        # delete snapshots, display video clips, delete video clips, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video filename to delete
            video_filename = input("Enter the video filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces functions `delete_video_clip` and `delete_all_video_clips` to delete individual video clips or all video clips, respectively.
- Press 'y' during real-time facial recognition to delete specific video clips or all video clips in the `VIDEO_DIR`.
- The user is prompted to enter the video filename to delete or enter 'all' to delete all video clips.

Feel free to run the updated script and explore the new video clip management features!




Certainly! Let's continue by implementing the option to display a live stream of the saved video clips. This feature will allow you to view the video clips in real-time, providing additional functionality to the video clip management system.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write(frame)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to delete a specific video clip
def delete_video_clip(video_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    try:
        os.remove(video_path)
        print(f"Video clip deleted: {video_filename}")
    except FileNotFoundError:
        print(f"Video clip not found: {video_filename}")

# Function to delete all video clips in the video directory
def delete_all_video_clips():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        os.remove(video_path)
    print("All video clips deleted.")

# Function to display a live stream of the saved video clips
def display_live_video_stream():
    while True

:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Loop through each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Live Video Stream", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the live video stream
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, display snapshots,
        # delete snapshots, display video clips, delete video clips, display live video stream, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video filename to delete
            video_filename = input("Enter the video filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        elif key == ord('l'):
            # Display live video stream of saved video clips
            display_live_video_stream()

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces the function `display_live_video_stream` to display a live stream of the saved video clips.
- Press 'l' during real-time facial recognition to display a live video stream of the saved video clips.

Feel free to run the updated script and explore the new live video stream feature!




Certainly! Let's continue by implementing the ability to delete individual snapshots and all snapshots through user input. This will enhance the snapshot management features of the script.

```python
import face_recognition
import cv2
import pickle
import os
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write(frame)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to delete a specific video clip
def delete_video_clip(video_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    try:
        os.remove(video_path)
        print(f"Video clip deleted: {video_filename}")
    except FileNotFoundError:
        print(f"Video clip not found: {video_filename}")

# Function to delete all video clips in the video directory
def delete_all_video_clips():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        os.remove(video_path)
    print("All video clips deleted.")

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove

(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to display a live stream of the saved video clips
def display_live_video_stream():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Loop through each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Live Video Stream", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the live video stream
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, display snapshots,
        # delete snapshots, display video clips, delete video clips, display live video stream, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video filename to delete
            video_filename = input("Enter the video filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        elif key == ord('l'):
            # Display live video stream of saved video clips
            display_live_video_stream()

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close
```





Certainly! Let's continue by adding a feature to export the recognized faces and their encodings to a CSV file. This functionality will allow you to save the known faces data and load it later for future use.

```python
import face_recognition
import cv2
import pickle
import os
import csv
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# CSV file to export known faces and their encodings
KNOWN_FACES_CSV = "known_faces.csv"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to export known faces and their encodings to a CSV file
def export_known_faces_to_csv(known_face_encodings, known_face_names):
    with open(KNOWN_FACES_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'Encoding'])

        for name, encoding in zip(known_face_names, known_face_encodings):
            csv_writer.writerow([name, encoding])

    print(f"Known faces exported to {KNOWN_FACES_CSV}.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write(frame)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to delete a specific video clip
def delete_video_clip(video_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    try:
   
```






Certainly! Let's continue by adding a feature to export the recognized faces and their encodings to a CSV file. This functionality will allow you to save the known faces data and load it later for future use.

```python
import face_recognition
import cv2
import pickle
import os
import csv
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# CSV file to export known faces and their encodings
KNOWN_FACES_CSV = "known_faces.csv"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to export known faces and their encodings to a CSV file
def export_known_faces_to_csv(known_face_encodings, known_face_names):
    with open(KNOWN_FACES_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'Encoding'])

        for name, encoding in zip(known_face_names, known_face_encodings):
            csv_writer.writerow([name, encoding])

    print(f"Known faces exported to {KNOWN_FACES_CSV}.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write(frame)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to delete a specific video clip
def delete_video_clip(video_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    try:
   

 os.remove(video_path)
        print(f"Video clip deleted: {video_filename}")
    except FileNotFoundError:
        print(f"Video clip not found: {video_filename}")

# Function to delete all video clips in the video directory
def delete_all_video_clips():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        os.remove(video_path)
    print("All video clips deleted.")

# Function to display a live stream of the saved video clips
def display_live_video_stream():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Loop through each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Live Video Stream", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the live video stream
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2
```


```python
.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, export faces to CSV,
        # display snapshots, delete snapshots, display video clips, delete video clips,
        # display live video stream, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('e'):
            # Export known faces and their encodings to CSV
            export_known_faces_to_csv(known_face_encodings, known_face_names)

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video filename to delete
            video_filename = input("Enter the video filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        elif key == ord('l'):
            # Display live video stream of saved video clips
            display_live_video_stream()

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings
    known_face_encodings, known_face_names = load_known_faces()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces the option to export known faces and their encodings to a CSV file (`export_known_faces_to_csv` function). Press 'e' during real-time facial recognition to export the known faces data to a CSV file named `known_faces.csv`.
- The user can choose to export known faces data to a CSV file, making it easy to reload this data in future runs of the script.

Feel free to run the updated script and explore the new export feature!





Certainly! Let's add the ability to load known faces and their encodings from a CSV file. This feature will enable you to start with a pre-existing set of known faces when running the script.

```python
import face_recognition
import cv2
import pickle
import os
import csv
from datetime import datetime

# File to store known faces and their encodings
KNOWN_FACES_FILE = "known_faces.pkl"
# CSV file to export known faces and their encodings
KNOWN_FACES_CSV = "known_faces.csv"
# Directory to save snapshots
SNAPSHOT_DIR = "snapshots"
# Directory to save video clips
VIDEO_DIR = "video_clips"

# Function to create the snapshot directory if it doesn't exist
def create_snapshot_directory():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

# Function to create the video directory if it doesn't exist
def create_video_directory():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

# Function to load known faces and their encodings from a file
def load_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'rb') as file:
            known_faces_data = pickle.load(file)
        return known_faces_data['encodings'], known_faces_data['names']
    except FileNotFoundError:
        return [], []

# Function to save known faces and their encodings to a file
def save_known_faces(known_face_encodings, known_face_names):
    known_faces_data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(KNOWN_FACES_FILE, 'wb') as file:
        pickle.dump(known_faces_data, file)

# Function to load known faces and their encodings from a CSV file
def load_known_faces_from_csv():
    known_face_encodings = []
    known_face_names = []

    try:
        with open(KNOWN_FACES_CSV, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row

            for row in csv_reader:
                name, encoding_str = row
                encoding = [float(value) for value in encoding_str.split(',')]
                known_face_encodings.append(encoding)
                known_face_names.append(name)

        print(f"Loaded {len(known_face_names)} faces from {KNOWN_FACES_CSV}.")
    except FileNotFoundError:
        print(f"{KNOWN_FACES_CSV} not found. No known faces loaded.")

    return known_face_encodings, known_face_names

# Function to add a new face to the known faces
def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)
    print(f"Added {new_face_name} to the known faces.")
    save_known_faces(known_face_encodings, known_face_names)

# Function to remove a face from the known faces
def remove_face(known_face_encodings, known_face_names, name_to_remove):
    if name_to_remove in known_face_names:
        index_to_remove = known_face_names.index(name_to_remove)
        known_face_encodings.pop(index_to_remove)
        known_face_names.pop(index_to_remove)
        print(f"Removed {name_to_remove} from the known faces.")
        save_known_faces(known_face_encodings, known_face_names)
    else:
        print(f"{name_to_remove} not found in the known faces.")

# Function to reset known faces, clearing all stored data
def reset_known_faces():
    try:
        os.remove(KNOWN_FACES_FILE)
        print("Reset known faces. All stored face data has been cleared.")
    except FileNotFoundError:
        print("No known faces data found. Nothing to reset.")

# Function to export known faces and their encodings to a CSV file
def export_known_faces_to_csv(known_face_encodings, known_face_names):
    with open(KNOWN_FACES_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'Encoding'])

        for name, encoding in zip(known_face_names, known_face_encodings):
            encoding_str = ','.join(map(str, encoding))
            csv_writer.writerow([name, encoding_str])

    print(f"Known faces exported to {KNOWN_FACES_CSV}.")

# Function to save a snapshot when a known face is recognized
def save_snapshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{SNAPSHOT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(snapshot_filename, frame)
    print(f"Snapshot saved: {snapshot_filename}")

# Function to display saved snapshots in a separate window
def display_snapshots():
    while True:
        # Get a list of snapshot files in the snapshot directory
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]

        if not snapshot_files:
            print("No snapshots found.")
            break

        # Display each snapshot in a window
        for snapshot_file in snapshot_files:
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            snapshot = cv2.imread(snapshot_path)

            # Display the snapshot
            cv2.imshow("Saved Snapshots", snapshot)
            cv2.waitKey(2000)  # Display each snapshot for 2 seconds

        # Break the loop after displaying all snapshots
        break

    cv2.destroyAllWindows()

# Function to delete a specific snapshot
def delete_snapshot(snapshot_filename):
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
    try:
        os.remove(snapshot_path)
        print(f"Snapshot deleted: {snapshot_filename}")
    except FileNotFoundError:
        print(f"Snapshot not found: {snapshot_filename}")

# Function to delete all snapshots in the snapshot directory
def delete_all_snapshots():
    snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
        os.remove(snapshot

_path)
    print("All snapshots deleted.")

# Function to save a video clip when a known face is recognized
def save_video_clip(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{VIDEO_DIR}/{name}_{timestamp}.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Write frames to the video file
    for _ in range(50):  # Capture 50 frames (2.5 seconds) for the video clip
        video_writer.write(frame)

    video_writer.release()
    print(f"Video clip saved: {video_filename}")

# Function to display saved video clips in a separate window
def display_video_clips():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Display each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Saved Video Clips", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the video clip display
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to delete a specific video clip
def delete_video_clip(video_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    try:
        os.remove(video_path)
        print(f"Video clip deleted: {video_filename}")
    except FileNotFoundError:
        print(f"Video clip not found: {video_filename}")

# Function to delete all video clips in the video directory
def delete_all_video_clips():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        os.remove(video_path)
    print("All video clips deleted.")

# Function to display a live stream of the saved video clips
def display_live_video_stream():
    while True:
        # Get a list of video files in the video directory
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]

        if not video_files:
            print("No video clips found.")
            break

        # Loop through each video clip in a window
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_capture = cv2.VideoCapture(video_path)

            while True:
                # Read each frame from the video clip
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Display the frame
                cv2.imshow("Live Video Stream", frame)

                # Break the loop after displaying the video clip
                if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit the live video stream
                    break

            video_capture.release()

        # Break the loop after displaying all video clips
        break

    cv2.destroyAllWindows()

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of
```





```python
            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({confidence:.2f})" if confidence else name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, export faces to CSV,
        # display snapshots, delete snapshots, display video clips, delete video clips,
        # display live video stream, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('e'):
            # Export known faces and their encodings to CSV
            export_known_faces_to_csv(known_face_encodings, known_face_names)

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video filename to delete
            video_filename = input("Enter the video filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        elif key == ord('l'):
            # Display live video stream of saved video clips
            display_live_video_stream()

        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load known faces and their encodings from a CSV file
    known_face_encodings, known_face_names = load_known_faces_from_csv()

    # Perform real-time facial recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script adds the ability to load known faces and their encodings from a CSV file (`load_known_faces_from_csv` function). Press 'l' during real-time facial recognition to display a live video stream of saved video clips.
- You can use the `KNOWN_FACES_CSV` file to maintain a persistent list of known faces across multiple script executions.

Feel free to run the updated script and explore the new load from CSV and live video stream features!




Certainly! Let's continue by adding a few more features to the script, including the ability to display live facial landmarks and face encodings. Additionally, we'll introduce an option to toggle displaying the names and confidence levels during real-time facial recognition.

```python
import face_recognition
import cv2
import pickle
import os
import csv
from datetime import datetime

# ... (previous code)

# Function to display facial landmarks on the frame
def display_facial_landmarks(frame, face_landmarks):
    for face_landmark in face_landmarks:
        for point in face_landmark.keys():
            for x, y in face_landmark[point]:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

# Function to display face encodings on the frame
def display_face_encodings(frame, face_encodings, face_locations):
    for face_encoding, face_location in zip(face_encodings, face_locations):
        top, right, bottom, left = face_location
        text = ", ".join(["{:.2f}".format(value) for value in face_encoding])
        cv2.putText(frame, text, (left + 6, bottom + 15), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

# Function to toggle displaying names and confidence during real-time facial recognition
def toggle_display_names():
    global display_names
    display_names = not display_names
    print(f"Displaying {'Names and Confidence' if display_names else 'Face Encodings'} during recognition.")

# Function to perform facial recognition on a video stream
def perform_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if faces are found
        if not face_locations:
            print("No faces found in the frame.")
            continue

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if display_names:
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{name} ({confidence:.2f})" if confidence else name
                cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                # Display face encodings instead of names and confidence
                display_face_encodings(frame, [face_encoding], [face_locations[0]])

        # Display facial landmarks if the 'l' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('l'):
            face_landmarks = face_recognition.face_landmarks(rgb_frame)
            display_facial_landmarks(frame, face_landmarks)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition", frame)

        # Check for user input to add, remove, reset faces, export faces to CSV,
        # display snapshots, delete snapshots, display video clips, delete video clips,
        # display live video stream, toggle display names, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('e'):
            # Export known faces and their encodings to CSV
            export_known_faces_to_csv(known_face_encodings, known_face_names)

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video filename to delete
            video_filename = input("Enter the video filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        elif key == ord('l'):
            # Display live video stream of saved video clips
            display_live_video_stream()

        elif key == ord('t'):
            # Toggle displaying names and confidence or face encodings
            toggle_display_names()

        elif key == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# ... (remaining code)

# Main function
def main():
    # Load known faces and their encodings from a CSV file
    known_face_encodings, known_face_names = load_known_faces_from_csv()

    # Perform real-time facial

 recognition on the webcam stream
    perform_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces the option to display live facial landmarks on the frame by pressing the 'l' key during real-time facial recognition.
- Press 't' during real-time facial recognition to toggle between displaying names and confidence levels or face encodings.
- The `display_facial_landmarks` function is added to draw facial landmarks on the frame.
- The `display_face_encodings` function is added to display face encodings on the frame.

Feel free to run the updated script and explore these new features!





Certainly! Let's add the ability to recognize and track faces in a video file instead of using a webcam. Additionally, we'll introduce the option to save the real-time facial recognition output as a video file.

```python
import face_recognition
import cv2
import pickle
import os
import csv
from datetime import datetime

# ... (previous code)

# Function to perform facial recognition on a video file
def perform_video_facial_recognition(video_path, known_face_encodings, known_face_names):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    create_snapshot_directory()
    create_video_directory()

    # Get the frames per second (fps) of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the output as a video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    output_video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (int(video_capture.get(3)), int(video_capture.get(4))))

    while True:
        # Capture each frame from the video
        ret, frame = video_capture.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if display_names:
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{name} ({confidence:.2f})" if confidence else name
                cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                # Display face encodings instead of names and confidence
                display_face_encodings(frame, [face_encoding], [face_locations[0]])

        # Write the frame to the output video file
        output_video_writer.write(frame)

        # Display the result in real-time
        cv2.imshow("Video Facial Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video file and output video writer, and close the window
    video_capture.release()
    output_video_writer.release()
    cv2.destroyAllWindows()

    print(f"Video facial recognition completed. Output saved as {output_filename}")

# Function to save the real-time facial recognition output as a video file
def save_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    # Get the frames per second (fps) of the webcam
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the output as a video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    output_video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (int(video_capture.get(3)), int(video_capture.get(4))))

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if display_names:
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{name} ({confidence:.2f})" if confidence else name
                cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                # Display face encodings instead of names and confidence
                display_face_encodings(frame, [face_encoding], [face_locations[0]])

        # Write the frame to the output video file
        output_video_writer.write(frame)

        # Display the result in real-time
        cv2

.imshow("Real-time Facial Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and output video writer, and close the window
    video_capture.release()
    output_video_writer.release()
    cv2.destroyAllWindows()

    print(f"Real-time facial recognition output saved as {output_filename}")

# ... (remaining code)

# Main function
def main():
    # Load known faces and their encodings from a CSV file
    known_face_encodings, known_face_names = load_known_faces_from_csv()

    # Perform real-time facial recognition on the webcam stream
    # perform_realtime_facial_recognition(known_face_encodings, known_face_names)

    # Uncomment the line above and comment the line below if you want to perform real-time facial recognition
    # Uncomment the line below and comment the line above if you want to perform facial recognition on a video file
    perform_video_facial_recognition('path/to/your/video.mp4', known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script adds the `perform_video_facial_recognition` function, allowing you to perform facial recognition on a video file. Update the path to your video file in the `perform_video_facial_recognition` function call.
- The script introduces the `save_realtime_facial_recognition` function, which captures the real-time facial recognition output and saves it as a video file. Press 'q' to stop the process and save the output.

Feel free to run the updated script and explore these new features!




Certainly! In this continuation, I'll introduce a feature to display the real-time facial recognition output with the option to save the video. Additionally, I'll provide a utility function to delete the last saved video file.

```python
import face_recognition
import cv2
import os
from datetime import datetime

# ... (previous code)

# Function to delete the last saved video file
def delete_last_saved_video():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    if video_files:
        latest_video_file = max(video_files, key=os.path.getctime)
        delete_video_clip(latest_video_file)
        print(f"Last saved video file '{latest_video_file}' deleted.")
    else:
        print("No video files found.")

# ... (remaining code)

# Function to display the real-time facial recognition output and save as a video file
def display_and_save_realtime_facial_recognition(known_face_encodings, known_face_names):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    create_snapshot_directory()
    create_video_directory()

    # Get the frames per second (fps) of the webcam
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the output as a video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    output_video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (int(video_capture.get(3)), int(video_capture.get(4))))

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"  # Default to unknown if no match is found
            confidence = None

            # If a match is found, use the name of the known face with the highest confidence
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

                # Save a snapshot and video clip when a known face is recognized
                save_snapshot(frame, name)
                save_video_clip(frame, name)

            # Draw a rectangle around the face and display the name and confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if display_names:
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{name} ({confidence:.2f})" if confidence else name
                cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                # Display face encodings instead of names and confidence
                display_face_encodings(frame, [face_encoding], [face_locations[0]])

        # Write the frame to the output video file
        output_video_writer.write(frame)

        # Display the result in real-time
        cv2.imshow("Real-time Facial Recognition and Save", frame)

        # Check for user input to add, remove, reset faces, export faces to CSV,
        # display snapshots, delete snapshots, display video clips, delete video clips,
        # toggle display names, delete last saved video, or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Capture a frame for the new face
            new_face_frame = frame[top:bottom, left:right]
            new_face_encoding = face_recognition.face_encodings(new_face_frame)[0]

            # Prompt user for the name of the new face
            new_face_name = input("Enter the name for the new face: ")

            # Add the new face to the known faces
            add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name)

        elif key == ord('r'):
            # Prompt user for the name to remove
            name_to_remove = input("Enter the name to remove: ")

            # Remove the specified face from the known faces
            remove_face(known_face_encodings, known_face_names, name_to_remove)

        elif key == ord('c'):
            # Reset known faces, clearing all stored data
            reset_known_faces()

        elif key == ord('e'):
            # Export known faces and their encodings to CSV
            export_known_faces_to_csv(known_face_encodings, known_face_names)

        elif key == ord('d'):
            # Display saved snapshots
            display_snapshots()

        elif key == ord('v'):
            # Display saved video clips
            display_video_clips()

        elif key == ord('x'):
            # Prompt user for the snapshot filename to delete
            snapshot_filename = input("Enter the snapshot filename to delete (or 'all' to delete all): ")

            if snapshot_filename.lower() == 'all':
                # Delete all snapshots
                delete_all_snapshots()
            else:
                # Delete a specific snapshot
                delete_snapshot(snapshot_filename)

        elif key == ord('y'):
            # Prompt user for the video filename to delete
            video_filename = input("Enter the video filename to delete (or 'all' to delete all): ")

            if video_filename.lower() == 'all':
                # Delete all video clips
                delete_all_video_clips()
            else:
                # Delete a specific video clip
                delete_video_clip(video_filename)

        elif key == ord('t'):
            # Toggle displaying names and confidence or face encodings
            toggle_display_names()

        elif key == ord('l'):
            # Delete the last saved video file
            delete_last_saved_video()

        elif key == ord('q'):
            break

    # Release the webcam and output video writer, and close the window
    video_capture.release()
    output_video_writer.release()
    cv2.destroyAllWindows()

    print(f"Real-time facial recognition output saved as {output_filename}")

# ... (remaining code)

# Main function
def main():
    # Load known faces and their encodings from a CSV file
    known_face_encodings, known_face_names = load_known_faces_from_csv()

    # Uncomment the line below and comment the line above if you want to perform real-time facial recognition
    display_and_save_realtime_facial_recognition(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
```

In this continuation:

- The script introduces the `delete_last_saved_video` function, allowing you to delete the last saved video file by pressing 'l'.
- The script adds the

`display_and_save_realtime_facial_recognition` function, which displays the real-time facial recognition output and saves it as a video file. Press 'q' to stop the process and save the output.

Feel free to run the updated script and explore these new features!
