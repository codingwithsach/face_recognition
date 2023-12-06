Certainly! Image processing with OpenCV involves several steps, and I'll provide you with a simple example in Python using OpenCV. Before running the code, make sure you have OpenCV installed. You can install it using:

```bash
pip install opencv-python
```

Now, let's go through a simple example step by step:

1. **Import OpenCV:**

   ```python
   import cv2
   ```
2. **Read an Image:**

   ```python
   # Replace 'path/to/your/image.jpg' with the path to your image file
   image = cv2.imread('path/to/your/image.jpg')
   ```
3. **Display the Original Image:**

   ```python
   cv2.imshow('Original Image', image)
   cv2.waitKey(0)  # Wait for a key press
   cv2.destroyAllWindows()  # Close the window
   ```
4. **Convert Image to Grayscale:**

   ```python
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   cv2.imshow('Grayscale Image', gray_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
5. **Apply Gaussian Blur:**

   ```python
   blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
   cv2.imshow('Blurred Image', blurred_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
6. **Edge Detection using Canny:**

   ```python
   edges = cv2.Canny(blurred_image, 50, 150)
   cv2.imshow('Edge Detection', edges)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
7. **Dilate and Erode:**

   ```python
   # Create a kernel for dilation and erosion
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

   # Dilate the image
   dilated_image = cv2.dilate(edges, kernel, iterations=1)

   # Erode the image
   eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

   cv2.imshow('Dilated Image', dilated_image)
   cv2.imshow('Eroded Image', eroded_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
8. **Save the Processed Image:**

   ```python
   # Save the final processed image
   cv2.imwrite('path/to/your/processed/image.jpg', eroded_image)
   ```

Feel free to adjust the parameters and explore more OpenCV functions based on your requirements. This example provides a basic introduction to common image processing operations.



Certainly! Let's continue with more image processing techniques using OpenCV:

9. **Contours:**

   ```python
   # Find contours in the eroded image
   contours, hierarchy = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # Draw contours on a blank image
   contour_image = image.copy()
   cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

   cv2.imshow('Contours', contour_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
10. **Histogram Equalization:**

    ```python
    # Convert the grayscale image back to BGR for equalization
    equalization_input = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Apply histogram equalization to the intensity channel
    equalized_image = cv2.equalizeHist(equalization_input)

    cv2.imshow('Histogram Equalization', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
11. **Resize and Crop:**

    ```python
    # Resize the image to a specific width and height
    new_width, new_height = 300, 200
    resized_image = cv2.resize(image, (new_width, new_height))

    # Crop a region of interest (ROI)
    x, y, w, h = 100, 50, 200, 150
    cropped_image = image[y:y + h, x:x + w]

    cv2.imshow('Resized Image', resized_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
12. **Rotation:**

    ```python
    # Get the image center and rotation matrix
    center = (image.shape[1] // 2, image.shape[0] // 2)
    angle = 45  # Specify the rotation angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

These additional steps cover a range of image processing techniques, from contour detection to histogram equalization, resizing, cropping, and rotation. Feel free to experiment with these techniques and combine them to suit your specific image processing needs.





Certainly! Let's continue with more image processing techniques using OpenCV:

9. **Contours:**

   ```python
   # Find contours in the eroded image
   contours, hierarchy = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # Draw contours on a blank image
   contour_image = image.copy()
   cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

   cv2.imshow('Contours', contour_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
10. **Histogram Equalization:**

    ```python
    # Convert the grayscale image back to BGR for equalization
    equalization_input = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Apply histogram equalization to the intensity channel
    equalized_image = cv2.equalizeHist(equalization_input)

    cv2.imshow('Histogram Equalization', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
11. **Resize and Crop:**

    ```python
    # Resize the image to a specific width and height
    new_width, new_height = 300, 200
    resized_image = cv2.resize(image, (new_width, new_height))

    # Crop a region of interest (ROI)
    x, y, w, h = 100, 50, 200, 150
    cropped_image = image[y:y + h, x:x + w]

    cv2.imshow('Resized Image', resized_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
12. **Rotation:**

    ```python
    # Get the image center and rotation matrix
    center = (image.shape[1] // 2, image.shape[0] // 2)
    angle = 45  # Specify the rotation angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

These additional steps cover a range of image processing techniques, from contour detection to histogram equalization, resizing, cropping, and rotation. Feel free to experiment with these techniques and combine them to suit your specific image processing needs.





Certainly! Let's continue with some more image processing techniques using OpenCV:

13. **Color Spaces:**

    ```python
    # Convert the BGR image to different color spaces (e.g., HSV, LAB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    cv2.imshow('HSV Image', hsv_image)
    cv2.imshow('LAB Image', lab_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
14. **Template Matching:**

    ```python
    # Load a template image
    template = cv2.imread('path/to/your/template.jpg')

    # Perform template matching
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw a rectangle around the matched region
    top_left = max_loc
    h, w = template.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('Template Matching', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
15. **Image Filtering (Custom Kernels):**

    ```python
    # Define a custom kernel (e.g., sharpening)
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])

    # Apply the custom kernel using convolution
    sharpened_image = cv2.filter2D(image, -1, kernel_sharpening)

    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
16. **Face Detection using Haarcascades:**

    ```python
    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

Feel free to explore and experiment with these additional techniques. OpenCV provides a wide range of functions for image processing, computer vision, and machine learning tasks, making it a powerful library for various applications.





Certainly! Let's continue with some more advanced image processing techniques using OpenCV:

17. **Image Thresholding:**

    ```python
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binary Thresholding', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
18. **Image Smoothing (Blurring):**

    ```python
    # Apply a Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

    cv2.imshow('Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
19. **Background Subtraction:**

    ```python
    # Create a background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Apply background subtraction
    foreground_mask = bg_subtractor.apply(image)

    cv2.imshow('Foreground Mask', foreground_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
20. **Image Morphology (Opening and Closing):**

    ```python
    # Create a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Perform morphological opening (erosion followed by dilation)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Perform morphological closing (dilation followed by erosion)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Morphological Opening', opening)
    cv2.imshow('Morphological Closing', closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
21. **Hough Transform for Line Detection:**

    ```python
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # Apply Hough Transform for line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

    # Draw lines on the image
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Hough Transform - Line Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

Feel free to explore and experiment with these advanced image processing techniques. OpenCV is a versatile library with extensive documentation, so you can always refer to the official documentation for more details and options: https://docs.opencv.org/




Certainly! Let's explore a few more advanced image processing techniques using OpenCV:

22. **Hough Transform for Circle Detection:**

    ```python
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

    # Apply Hough Transform for circle detection
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)

    # Draw circles on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

    cv2.imshow('Hough Transform - Circle Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
23. **Image Stitching (Panorama):**

    ```python
    # Load two images
    image1 = cv2.imread('path/to/your/image1.jpg')
    image2 = cv2.imread('path/to/your/image2.jpg')

    # Detect keypoints and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Match descriptors using a brute force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to obtain good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw the matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Image Stitching', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
24. **Object Tracking (Lucas-Kanade Optical Flow):**

    ```python
    # Capture video from a webcam
    cap = cv2.VideoCapture(0)

    # Read the first frame
    ret, frame = cap.read()
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the current frame
        ret, frame = cap.read()
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Update the previous frame
        prev_frame = current_frame

        # Draw the optical flow vectors
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        frame_with_flow = cv2.polylines(frame, lines, 0, (0, 255, 0))

        cv2.imshow('Optical Flow', frame_with_flow)

        # Break the loop on 'Esc' key press
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

These examples demonstrate more advanced image processing techniques, including circle detection, image stitching for creating panoramas, and optical flow for object tracking. Feel free to adapt and experiment with these techniques based on your specific needs.





Certainly! Let's delve into a few more advanced image processing and computer vision techniques using OpenCV:

25. **Foreground Extraction using GrabCut:**

    ```python
    # Load an image with an object and a rough mask
    image = cv2.imread('path/to/your/image.jpg')
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define a rectangle around the object to initialize GrabCut
    rect = (50, 50, 450, 290)
    cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to obtain the foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]

    cv2.imshow('Foreground Extraction', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
26. **Histogram of Oriented Gradients (HOG) for Object Detection:**

    ```python
    # Load an image and a pre-trained HOG descriptor
    image = cv2.imread('path/to/your/image.jpg')
    hog = cv2.HOGDescriptor()

    # Detect HOG features in the image
    features, _ = hog.compute(image)

    # Visualize HOG features
    hog_image = hog.render(features)

    cv2.imshow('HOG Features', hog_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
27. **Convolutional Neural Networks (CNN) for Image Classification using OpenCV and DNN module:**

    ```python
    # Load a pre-trained CNN model (e.g., MobileNet)
    net = cv2.dnn.readNetFromTensorflow('path/to/your/model.pb', 'path/to/your/config.pbtxt')

    # Load and preprocess an image for classification
    image = cv2.imread('path/to/your/image.jpg')
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(224, 224), mean=(104, 117, 123), swapRB=True)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass to obtain predictions
    predictions = net.forward()

    # Get the class label with the highest probability
    class_id = np.argmax(predictions[0])
    confidence = predictions[0][class_id]

    # Display the result
    label = f"Class: {class_id}, Confidence: {confidence:.2f}"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Image Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

These examples cover more advanced topics, such as foreground extraction using GrabCut, Histogram of Oriented Gradients (HOG) for object detection, and using Convolutional Neural Networks (CNN) for image classification with the OpenCV DNN module. Experimenting with these techniques will provide you with a deeper understanding of image processing and computer vision.





Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

28. **Mean Shift and CamShift for Object Tracking:**

    ```python
    # Capture video from a webcam
    cap = cv2.VideoCapture(0)

    # Read the first frame
    ret, frame = cap.read()

    # Set up the initial tracking window
    roi = cv2.selectROI(frame)
    track_window = roi

    # Set up the termination criteria
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Apply mean shift algorithm
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        dst = cv2.calcBackProject([hsv], [0], mask, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw a rectangle around the tracked object
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

        cv2.imshow('Mean Shift Tracking', img2)

        # Break the loop on 'Esc' key press
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ```
29. **SIFT (Scale-Invariant Feature Transform) for Feature Matching:**

    ```python
    # Load two images
    image1 = cv2.imread('path/to/your/image1.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('path/to/your/image2.jpg', cv2.IMREAD_GRAYSCALE)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Match descriptors using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches on the images
    result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('SIFT Feature Matching', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
30. **Depth Map Estimation using Stereo Vision:**

    ```python
    # Capture video from stereo cameras
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    while True:
        # Read frames from both cameras
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        # Convert frames to grayscale
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # Perform stereo block matching to compute disparity map
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(gray_left, gray_right)

        # Normalize the disparity map for display
        normalized_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

        # Display the disparity map
        cv2.imshow('Disparity Map', normalized_disparity)

        # Break the loop on 'Esc' key press
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    ```

These examples cover mean shift and CamShift for object tracking, SIFT for feature matching, and depth map estimation using stereo vision. Each of these techniques involves more advanced concepts and applications, providing a solid foundation for diving deeper into computer vision and image processing.





Certainly! Let's explore a few more advanced techniques in image processing and computer vision using OpenCV:

31. **Facial Landmark Detection with Dlib:**

    ```python
    # Install dlib library (if not installed): pip install dlib
    import dlib

    # Load the pre-trained facial landmark predictor
    predictor_path = 'path/to/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Load an image
    image = cv2.imread('path/to/your/image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Loop over each face and detect facial landmarks
    for face in faces:
        landmarks = predictor(gray, face)

        # Draw facial landmarks on the image
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Facial Landmark Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
32. **Image Segmentation using GrabCut with Rectangular and Freehand Masks:**

    ```python
    # Load an image
    image = cv2.imread('path/to/your/image.jpg')

    # Define a rectangle or create a freehand ROI for GrabCut
    rect = (50, 50, 450, 290)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask using a freehand drawing
    mask2 = cv2.imread('path/to/your/freehand_mask.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.grabCut(image, mask2, None, None, None, 5, cv2.GC_INIT_WITH_MASK)

    # Modify the mask to obtain the foreground
    mask_result = np.where((mask2 == 2) | (mask2 == 0), 0, 1).astype('uint8')
    result = image * mask_result[:, :, np.newaxis]

    cv2.imshow('GrabCut with Mask', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
33. **Hand Gesture Recognition using Convex Hull:**

    ```python
    # Capture video from a webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur and thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the maximum area (hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Convex hull for the hand contour
        hull = cv2.convexHull(max_contour)

        # Draw the hand contour and convex hull
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        # Break the loop on 'Esc' key press
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

These examples cover facial landmark detection using Dlib, image segmentation using GrabCut with rectangular and freehand masks, and hand gesture recognition using convex hull. Each of these techniques demonstrates different aspects of computer vision and image processing, allowing you to explore diverse applications.





Certainly! Let's continue exploring some more advanced image processing and computer vision techniques using OpenCV:

34. **Image Recognition with Transfer Learning (MobileNetV2):**

    ```python
    # Load the pre-trained MobileNetV2 model
    net = cv2.dnn.readNetFromTensorflow('path/to/mobilenet_v2.pb')

    # Load and preprocess an image for classification
    image = cv2.imread('path/to/your/image.jpg')
    blob = cv2.dnn.blobFromImage(image, size=(224, 224), swapRB=True, crop=False)
    net.setInput(blob)

    # Make predictions and get class labels
    predictions = net.forward()
    class_id = np.argmax(predictions[0])
    confidence = predictions[0][class_id]

    # Display the result
    label = f"Class: {class_id}, Confidence: {confidence:.2f}"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Image Recognition (MobileNetV2)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
35. **Document Scanner using Perspective Transform:**

    ```python
    # Load an image of a document
    image = cv2.imread('path/to/your/document.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (document)
    max_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # Apply perspective transform to obtain a top-down view
    warped = four_point_transform(image, approx.reshape(4, 2))

    cv2.imshow('Document Scanner', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
36. **Real-Time Object Detection using YOLO (You Only Look Once):**

    ```python
    # Load YOLO model
    net = cv2.dnn.readNet('path/to/yolov3.weights', 'path/to/yolov3.cfg')

    # Load COCO names for classes
    with open('path/to/coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    # Load an image and get its height and width
    image = cv2.imread('path/to/your/image.jpg')
    height, width = image.shape[:2]

    # Create a blob from the image and perform forward pass
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections and draw bounding boxes
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(obj[0] * width), int(obj[1] * height)
                w, h = int(obj[2] * width), int(obj[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f'{classes[class_id]}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Real-Time Object Detection (YOLO)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

These examples cover image recognition with transfer learning using MobileNetV2, document scanning using perspective transform, and real-time object detection using YOLO. Each of these techniques showcases the versatility of OpenCV in handling various computer vision tasks.





Absolutely! Let's continue exploring some more advanced image processing and computer vision techniques using OpenCV:

37. **Augmented Reality with ArUco Markers:**

    ```python
    # Install the `opencv-contrib-python` library (if not installed): pip install opencv-contrib-python
    import cv2
    import cv2.aruco as aruco
    import numpy as np

    # Create a dictionary and initialize the detector parameters
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # Capture video from a webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Draw the markers and annotate with IDs
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                cv2.putText(frame, f'ID: {ids[i][0]}', tuple(corners[i][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the augmented reality view
        cv2.imshow('Augmented Reality with ArUco Markers', frame)

        # Break the loop on 'Esc' key press
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ```
38. **Lane Detection in Road Scenes:**

    ```python
    # Capture video from a source (e.g., webcam or video file)
    cap = cv2.VideoCapture('path/to/your/video.mp4')

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Define region of interest (ROI) for lane detection
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[(0, frame.shape[0]), (frame.shape[1] // 2, frame.shape[0] // 2), (frame.shape[1], frame.shape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough Transform for line detection
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

        # Draw detected lines on the frame
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the lane detection result
        cv2.imshow('Lane Detection', frame)

        # Break the loop on 'Esc' key press
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ```
39. **Handwritten Digit Recognition using MNIST and Keras:**

    ```python
    # Install the `keras` library (if not installed): pip install keras
    from keras.models import load_model
    import cv2
    import numpy as np

    # Load the pre-trained MNIST digit recognition model
    model = load_model('path/to/mnist_model.h5')

    # Function to preprocess and predict the digit in an image
    def predict_digit(image):
        image = cv2.resize(image, (28, 28))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        digit = np.argmax(prediction)
        return digit

    # Load an image of a handwritten digit
    image = cv2.imread('path/to/your/digit_image.jpg')

    # Predict the digit in the image
    predicted_digit = predict_digit(image)

    # Display the result
    cv2.putText(image, f'Predicted Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Handwritten Digit Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

These examples cover augmented reality with

ArUco markers, lane detection in road scenes, and handwritten digit recognition using a pre-trained model and Keras. Each of these techniques showcases the practical applications of computer vision in different domains. Feel free to adapt and experiment with these examples for your specific use cases!





Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

40. **Real-Time Facial Expression Recognition:**

    ```python
    # Install the `dlib` library (if not installed): pip install dlib
    # Download the facial expression recognition model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # Extract the model and provide the path below

    import cv2
    import dlib
    import numpy as np

    # Load the pre-trained facial landmark predictor
    predictor_path = 'path/to/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Capture video from a webcam
    cap = cv2.VideoCapture(0)

    # Load the pre-trained facial expression recognition model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('path/to/emotion_model.xml')

    # Define emotions
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(gray, face)
            landmarks_np = np.array([[point.x, point.y] for point in landmarks.parts()])

            # Extract the region of interest (ROI) for facial expression recognition
            (x, y, w, h) = cv2.boundingRect(landmarks_np)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Predict the emotion
            emotion_id, confidence = model.predict(roi_gray)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Annotate the emotion on the frame
            emotion_text = f"Emotion: {emotions[emotion_id]} ({confidence:.2f})"
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the real-time facial expression recognition result
        cv2.imshow('Facial Expression Recognition', frame)

        # Break the loop on 'Esc' key press
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ```
41. **3D Object Detection with Point Clouds:**

    ```python
    # Install the `open3d` library (if not installed): pip install open3d
    import open3d as o3d
    import numpy as np

    # Create a point cloud from a depth map (replace 'depth_image.npy' with your depth map)
    depth_image = np.load('path/to/depth_image.npy')
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(depth_image), intrinsics)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    ```
42. **3D Object Reconstruction with Structure from Motion (SfM):**

    ```python
    # Install the `colmap` library (if not installed): https://colmap.github.io/install.html
    # Run COLMAP on your images to create a model, and specify the path to the model file below
    # For example, run: colmap automatic_reconstructor --image_path path/to/images --output_path path/to/output

    import colmap

    # Load the reconstructed 3D model
    model_path = 'path/to/your/colmap_model'
    model = colmap.read_model(model_path)

    # Visualize the 3D model
    colmap.show_model(model)
    ```

These examples cover real-time facial expression recognition, 3D object detection with point clouds, and 3D object reconstruction with Structure from Motion (SfM). These techniques demonstrate the diverse range of applications that can be achieved with advanced computer vision and image processing. Feel free to experiment with these examples and adapt them to your specific projects!





Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

43. **Object Tracking with SORT (Simple Online and Realtime Tracking):**

```python
# Install the `sort` library (if not installed): pip install sort
# Clone the SORT repository: git clone https://github.com/abewley/sort.git

from sort import Sort
import cv2

# Create a SORT tracker
tracker = Sort()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Perform object detection or use pre-trained model to obtain bounding boxes
    # For simplicity, let's assume we have bounding boxes in the 'detections' list

    # detections format: [x, y, width, height, confidence]
    detections = [[100, 100, 50, 50, 0.9], [200, 200, 40, 40, 0.8]]

    # Update the tracker with new detections
    trackers = tracker.update(np.array(detections))

    # Loop over the trackers and draw bounding boxes on the frame
    for tracking in trackers:
        x, y, w, h, track_id = tracking.astype(int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the object tracking result
    cv2.imshow('Object Tracking with SORT', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

44. **Semantic Segmentation using DeepLabV3:**

```python
# Install the `opencv-python` and `tensorflow` libraries (if not installed): pip install opencv-python tensorflow
# Download the DeepLabV3 model from: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
# Choose the appropriate DeepLabV3 model and provide the path below

import cv2
import numpy as np
import tensorflow as tf

# Load the DeepLabV3 model
model_path = 'path/to/deeplab_model/deeplabv3_mnv2_pascal_train_aug'
model = tf.saved_model.load(model_path)

# Create a function for semantic segmentation
def segment_image(image):
    input_tensor = tf.convert_to_tensor([image])
    output_dict = model(input_tensor)
    masks = output_dict['decoded_labels'][0].numpy()
    return masks

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Resize the frame to match the DeepLabV3 model input size
    input_size = (513, 513)
    resized_frame = cv2.resize(frame, input_size)

    # Perform semantic segmentation
    masks = segment_image(resized_frame)

    # Apply the segmentation masks to the original frame
    segmented_frame = frame.copy()
    for i in range(masks.shape[2]):
        mask = masks[:, :, i]
        color = np.random.randint(0, 255, size=3)
        segmented_frame[mask > 0] = color

    # Display the semantic segmentation result
    cv2.imshow('Semantic Segmentation with DeepLabV3', segmented_frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

45. **Image Style Transfer using Neural Style Transfer:**

```python
# Install the `opencv-python` and `torch` libraries (if not installed): pip install opencv-python torch torchvision
# Download the pre-trained VGG-19 model from: https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
# Download the pre-trained style transfer model from: https://pytorch.org/hub/pytorch_vision_vgg/

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# Load the VGG-19 model
vgg = models.vgg19()
vgg.load_state_dict(torch.load('path/to/vgg19-dcbb9e9d.pth'))
vgg = nn.Sequential(*list(vgg.features.children())[:-1]).eval()

# Load the pre-trained style transfer model
model = torch.hub.load('pytorch/vision:v0.10.0', 'fast_neural_style', pretrained=True)

# Define the style transfer function
def style_transfer(content_image, style_image, num_steps=300):
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    style_image = style_transform(style_image)

    content_image = content_image.unsqueeze(0).to(device='cuda', dtype=torch.float32)
    style_image = style_image.unsqueeze(0).to(device='cuda', dtype=torch.float32)

    input_image = content_image.clone()

    optimizer = optim.LBFGS([input_image.requires_grad_()])

    mse_loss = nn.MSELoss()

    run = [0]

    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()

            input_features = vgg(input_image)
            content_features = vgg(content_image)
            style_features = vgg(style_image)

            content_loss = mse_loss(input_features[1], content_features[1])

            style_loss = 0
            for j in range(len(style_features)):
                style_gram = torch.Tensor(style_features[j].shape[1], style_features[j].shape[1]).to(device='cuda', dtype=torch.float32)
                input_gram = torch.Tensor(style_features[j].shape[1], style_features[j].shape[1]).to(device='cuda', dtype=torch.float32)
                for k in range(style_features[j].shape[0]):
                    style_gram +=

torch.mm(style_features[j][0, k, :, :].view(style_features[j].shape[2] * style_features[j].shape[3], 1),
                                           style_features[j][0, k, :, :].view(style_features[j].shape[2] * style_features[j].shape[3], 1).t())
                    input_gram += torch.mm(input_features[j][0, k, :, :].view(input_features[j].shape[2] * input_features[j].shape[3], 1),
                                           input_features[j][0, k, :, :].view(input_features[j].shape[2] * input_features[j].shape[3], 1).t())

                style_loss += mse_loss(input_gram, style_gram)

            total_loss = content_loss + style_loss
            total_loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step [{run[0]}/{num_steps}], Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

            return total_loss

        optimizer.step(closure)

    return input_image.squeeze()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Load the style image
style_image = cv2.imread('path/to/style_image.jpg')

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Apply style transfer to the frame
    stylized_frame = style_transfer(frame, style_image)

    # Display the stylized frame
    cv2.imshow('Image Style Transfer', stylized_frame.astype(np.uint8))

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover object tracking with SORT, semantic segmentation using DeepLabV3, and image style transfer using neural style transfer. Feel free to experiment with these examples and adapt them to your specific projects!




Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

46. **Person Re-identification with OpenCV and Siamese Networks:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Load pre-trained Haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load two images for person re-identification
image1 = cv2.imread('path/to/person1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('path/to/person2.jpg', cv2.IMREAD_GRAYSCALE)

# Detect faces in the images
faces1 = face_cascade.detectMultiScale(image1, scaleFactor=1.3, minNeighbors=5)
faces2 = face_cascade.detectMultiScale(image2, scaleFactor=1.3, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces1:
    cv2.rectangle(image1, (x, y), (x+w, y+h), (255, 0, 0), 2)

for (x, y, w, h) in faces2:
    cv2.rectangle(image2, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the images with detected faces
cv2.imshow('Person 1', image1)
cv2.imshow('Person 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

47. **Image-to-Image Translation with Pix2Pix:**

```python
# Install the `tensorflow` and `opencv-python` libraries (if not installed): pip install tensorflow opencv-python

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the Pix2Pix model
model = load_model('path/to/pix2pix_model.h5')

# Load an image for image-to-image translation
input_image = cv2.imread('path/to/input_image.jpg')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (256, 256))
input_image = input_image / 255.0

# Perform image-to-image translation
generated_image = model.predict(np.expand_dims(input_image, axis=0))[0]

# Display the input and generated images
cv2.imshow('Input Image', cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
cv2.imshow('Generated Image', cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

48. **Real-Time Hand Pose Estimation with MediaPipe:**

```python
# Install the `mediapipe` library (if not installed): pip install mediapipe

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for point in landmarks.landmark:
                height, width, _ = frame.shape
                cx, cy = int(point.x * width), int(point.y * height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Display the hand pose estimation result
    cv2.imshow('Hand Pose Estimation', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover person re-identification with OpenCV, image-to-image translation with Pix2Pix, and real-time hand pose estimation with MediaPipe. Feel free to experiment with these examples and adapt them to your specific projects!





Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

49. **Real-Time Face Mesh Detection with MediaPipe:**

```python
# Install the `mediapipe` library (if not installed): pip install mediapipe

import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe FaceMesh
    results = face_mesh.process(rgb_frame)

    # Draw face landmarks on the frame
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for point in landmarks.landmark:
                height, width, _ = frame.shape
                cx, cy = int(point.x * width), int(point.y * height)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

    # Display the face mesh detection result
    cv2.imshow('Face Mesh Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

50. **Object Recognition using EfficientDet:**

```python
# Install the `opencv-python`, `tensorflow`, and `efficientdet` libraries (if not installed): pip install opencv-python tensorflow efficientdet

import cv2
import numpy as np
from efficientdet import EfficientDet

# Load the EfficientDet model
model = EfficientDet()

# Load an image for object recognition
image = cv2.imread('path/to/object_recognition_image.jpg')

# Perform object recognition
detections = model.predict(image)

# Draw bounding boxes on the image
for detection in detections:
    box = detection['box']
    class_name = detection['class_name']
    confidence = detection['confidence']

    color = (0, 255, 0)  # Green color for bounding boxes
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
    cv2.putText(image, f'{class_name}: {confidence:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the object recognition result
cv2.imshow('Object Recognition with EfficientDet', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

51. **Human Pose Estimation using OpenPose:**

```python
# Install the `openpose` library (if not installed): https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md

import cv2
import openpose as op

# Initialize OpenPose
params = {
    "model_folder": "path/to/openpose/models/",
    "hand": False,
    "face": False
}
openpose = op.WrapperPython()
openpose.configure(params)
openpose.start()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Process the frame with OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    openpose.emplaceAndPop([datum])

    # Draw human pose keypoints on the frame
    keypoints = datum.poseKeypoints
    if keypoints is not None:
        for person_keypoints in keypoints:
            for keypoint in person_keypoints:
                if keypoint[2] > 0.1:  # Confidence threshold
                    cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (0, 255, 0), -1)

    # Display the human pose estimation result
    cv2.imshow('Human Pose Estimation with OpenPose', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover real-time face mesh detection with MediaPipe, object recognition using EfficientDet, and human pose estimation using OpenPose. Feel free to experiment with these examples and adapt them to your specific projects!




Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

52. **Text Detection and Recognition using EAST and Tesseract:**

```python
# Install the `opencv-python`, `pytesseract`, and `imutils` libraries (if not installed): pip install opencv-python pytesseract imutils
# Make sure you have Tesseract installed on your system: https://github.com/tesseract-ocr/tesseract

import cv2
import pytesseract
from imutils.object_detection import non_max_suppression

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the EAST text detector
net = cv2.dnn.readNet('path/to/east_model.pb')

# Load an image for text detection and recognition
image = cv2.imread('path/to/text_image.jpg')

# Resize the image and get its dimensions
(H, W) = image.shape[:2]
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

# Preprocess the image for text detection
blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
scores, geometry = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

# Decode the predictions and apply non-maxima suppression
rectangles, confidences = non_max_suppression(scores, geometry)

# Iterate over the rectangles and extract text regions
for (startX, startY, endX, endY), confidence in zip(rectangles, confidences):
    # Scale the bounding box coordinates based on the original dimensions
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # Crop the text region from the image
    text_region = image[startY:endY, startX:endX]

    # Perform OCR on the text region using Tesseract
    text = pytesseract.image_to_string(text_region, config='--psm 6')

    # Draw the bounding box and recognized text on the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Display the text detection and recognition result
cv2.imshow('Text Detection and Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

53. **Camera Calibration and Undistortion:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Load a chessboard image for camera calibration
chessboard_image = cv2.imread('path/to/chessboard_image.jpg')
gray_chessboard = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)

# Define the number of inner corners in the chessboard (columns and rows)
chessboard_size = (9, 6)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray_chessboard, chessboard_size, None)

# If corners are found, refine the corners and calibrate the camera
if ret:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray_chessboard, corners, (11, 11), (-1, -1), criteria)
    objpoints = []
    imgpoints = []

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Store object points and image points for calibration
    objpoints.append(objp)
    imgpoints.append(corners)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_chessboard.shape[::-1], None, None)

    # Undistort an example image using the calibrated camera parameters
    distorted_image = cv2.imread('path/to/distorted_image.jpg')
    undistorted_image = cv2.undistort(distorted_image, mtx, dist, None, mtx)

    # Display the original and undistorted images
    cv2.imshow('Original Image', distorted_image)
    cv2.imshow('Undistorted Image', undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

54. **Image Stitching for Panorama Creation:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Load a set of images for panorama stitching
image1 = cv2.imread('path/to/image1.jpg')
image2 = cv2.imread('path/to/image2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors using ORB
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Create a Brute Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort them in ascending order of distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches
match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow('Matches', match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

These examples cover text detection and recognition using EAST and Tesseract, camera calibration and undistortion, and image stitching for panorama creation. Feel free to experiment with these examples and adapt them to your specific projects!





Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

55. **Super-Resolution with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Load a low-resolution image for super-resolution
lr_image = cv2.imread('path/to/low_resolution_image.jpg')

# Use OpenCV's super-resolution module to perform super-resolution
sr_image = cv2.detailEnhance(lr_image, sigma_s=10, sigma_r=0.15)

# Display the low-resolution and super-resolution images
cv2.imshow('Low-Resolution Image', lr_image)
cv2.imshow('Super-Resolution Image', sr_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

56. **Real-Time Object Detection with YOLO (You Only Look Once):**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Load the YOLO model and configuration
net = cv2.dnn.readNet('path/to/yolov3.weights', 'path/to/yolov3.cfg')

# Load the COCO class labels
with open('path/to/coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load the image for object detection
image = cv2.imread('path/to/object_detection_image.jpg')
height, width = image.shape[:2]

# Create a blob from the image and perform forward pass
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getUnconnectedOutLayersNames()
detections = net.forward(layer_names)

# Iterate over the detections and draw bounding boxes
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # Calculate the bounding box coordinates
            center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)
            x, y = int(center_x - w / 2), int(center_y - h / 2)

            # Draw the bounding box and label on the image
            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = f'{classes[class_id]}: {confidence:.2f}'
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the object detection result
cv2.imshow('Object Detection with YOLO', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

57. **Real-Time Facial Landmark Detection with Dlib:**

```python
# Install the `opencv-python` and `dlib` libraries (if not installed): pip install opencv-python dlib

import cv2
import dlib

# Load the pre-trained facial landmark predictor
predictor_path = 'path/to/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Draw facial landmarks on the frame
        for point in landmarks.parts():
            cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

    # Display the real-time facial landmark detection result
    cv2.imshow('Facial Landmark Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover super-resolution with OpenCV, real-time object detection with YOLO, and real-time facial landmark detection with Dlib. Feel free to experiment with these examples and adapt them to your specific projects!




Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

58. **Image Segmentation using U-Net:**

```python
# Install the `opencv-python` and `tensorflow` libraries (if not installed): pip install opencv-python tensorflow

import cv2
import numpy as np
import tensorflow as tf

# Load the U-Net model
model = tf.keras.models.load_model('path/to/unet_model.h5')

# Load an image for image segmentation
image = cv2.imread('path/to/image_for_segmentation.jpg')
input_image = cv2.resize(image, (256, 256))  # Resize the image to match the U-Net model input size
input_image = input_image / 255.0  # Normalize the image

# Perform image segmentation using U-Net
segmentation_mask = model.predict(np.expand_dims(input_image, axis=0))[0]
segmented_image = (segmentation_mask[:, :, 0] > 0.5).astype(np.uint8) * 255

# Display the original image and segmented result
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

59. **Real-Time Face Recognition using FaceNet:**

```python
# Install the `opencv-python`, `tensorflow`, and `face-recognition` libraries (if not installed): pip install opencv-python tensorflow face-recognition

import cv2
import face_recognition

# Load a known face image and encode it
known_image = face_recognition.load_image_file('path/to/known_face.jpg')
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    # Encode all face images in the frame
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Compare each face encoding with the known face encoding
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "Unknown"

        if matches[0]:
            name = "Known"

        # Draw the face rectangle and label on the frame
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the real-time face recognition result
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

60. **Real-Time Emotion Detection using CNN:**

```python
# Install the `opencv-python`, `tensorflow`, and `emotion` libraries (if not installed): pip install opencv-python tensorflow emotion

import cv2
from emotion import EmotionRecognition

# Initialize the EmotionRecognition model
emotion_recognition = EmotionRecognition(device='cpu')  # Use 'cuda' if a GPU is available

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Perform emotion detection on the frame
    emotions = emotion_recognition.predict(frame)

    # Display the emotion labels on the frame
    for emotion, (x, y, w, h) in emotions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the real-time emotion detection result
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover image segmentation using U-Net, real-time face recognition using FaceNet, and real-time emotion detection using a CNN. Feel free to experiment with these examples and adapt them to your specific projects!





Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

61. **Real-Time Lane Detection with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to the grayscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest
    height, width = edges.shape
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    roi_edges = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))

    # Use HoughLinesP to detect lines in the region of interest
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=30)

    # Draw lines on the frame
    draw_lines(frame, lines)

    # Display the real-time lane detection result
    cv2.imshow('Lane Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

62. **Real-Time Object Tracking with MOSSE Tracker:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Create a MOSSE tracker
tracker = cv2.TrackerMOSSE_create()

# Read the first frame from the video stream
ret, frame = cap.read()

# Select a region to track
bbox = cv2.selectROI('Object Tracking', frame, False)
tracker.init(frame, bbox)

while True:
    # Read a new frame from the video stream
    ret, frame = cap.read()

    # Update the tracker
    success, bbox = tracker.update(frame)

    # Draw the bounding box on the frame
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time object tracking result
    cv2.imshow('Object Tracking', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

63. **Real-Time Hand Gesture Recognition with MediaPipe Hands:**

```python
# Install the `mediapipe` library (if not installed): pip install mediapipe

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for point in landmarks.landmark:
                height, width, _ = frame.shape
                cx, cy = int(point.x * width), int(point.y * height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Display the hand gesture recognition result
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover real-time lane detection, real-time object tracking with MOSSE tracker, and real-time hand gesture recognition using MediaPipe Hands. Feel free to experiment with these examples and adapt them to your specific projects!





Certainly! Let's explore a few more advanced image processing and computer vision techniques using OpenCV:

64. **Image Style Transfer with Neural Style Transfer:**

```python
# Install the `opencv-python` and `tensorflow` libraries (if not installed): pip install opencv-python tensorflow

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

# Load the pre-trained VGG19 model
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def deprocess_image(x):
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = tf.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, 1:, :img_ncols-1, :])
    b = tf.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

def gram_matrix(x):
    features = tf.keras.backend.batch_flatten(tf.keras.backend.permute_dimensions(x, (2, 0, 1)))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_transfer(base_image_path, style_image_path, iterations=100):
    base_image = load_and_preprocess_image(base_image_path)
    style_image = load_and_preprocess_image(style_image_path)

    generated_image = tf.Variable(base_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=5.0)

    style_weight = 1.0
    content_weight = 0.025
    total_variation_weight = 1.0

    for i in range(iterations):
        with tf.GradientTape() as tape:
            loss = style_weight * style_loss(model(style_image), model(generated_image)) + \
                   content_weight * content_loss(model(base_image), model(generated_image)) + \
                   total_variation_weight * total_variation_loss(generated_image)

        gradients = tape.gradient(loss, generated_image)
        opt.apply_gradients([(gradients, generated_image)])

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy()}")

    result = deprocess_image(generated_image.numpy())
    return result

# Specify paths to the content and style images
content_image_path = 'path/to/content_image.jpg'
style_image_path = 'path/to/style_image.jpg'

# Set the dimensions for the generated image
img_nrows, img_ncols = 400, 400

# Perform style transfer
stylized_image = style_transfer(content_image_path, style_image_path)

# Display the content, style, and stylized images
cv2.imshow('Content Image', cv2.imread(content_image_path))
cv2.imshow('Style Image', cv2.imread(style_image_path))
cv2.imshow('Stylized Image', cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

65. **Real-Time Object Detection with EfficientDet and TensorFlow Lite:**

```python
# Install the `opencv-python`, `tensorflow`, and `tflite_runtime` libraries (if not installed): pip install opencv-python tensorflow tflite_runtime

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load the EfficientDet Lite model
model_path = 'path/to/efficientdet_lite_model.tflite'
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Resize the frame to match the model input size
    input_frame = cv2.resize(frame, (width, height))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame

, axis=0).astype(np.float32)

    # Set the input tensor for the model
    interpreter.set_tensor(input_details[0]['index'], input_frame)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_boxes = interpreter.get_tensor(output_details[0]['index'])
    output_classes = interpreter.get_tensor(output_details[1]['index'])
    output_scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    # Draw bounding boxes on the frame
    for i in range(num_detections):
        class_id = int(output_classes[0][i])
        score = float(output_scores[0][i])
        box = [float(val) for val in output_boxes[0][i]]

        if score > 0.5:
            x, y, w, h = int(box[1] * frame.shape[1]), int(box[0] * frame.shape[0]), \
                         int((box[3] - box[1]) * frame.shape[1]), int((box[2] - box[0]) * frame.shape[0])
            label = f'Class: {class_id}, Score: {score:.2f}'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the real-time object detection result
    cv2.imshow('Object Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover image style transfer using Neural Style Transfer, and real-time object detection with EfficientDet and TensorFlow Lite. Feel free to experiment with these examples and adapt them to your specific projects!




66. **Real-Time Document Scanner using OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def find_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            document_contour = approx
            break

    return document_contour

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Find the document contour
    document_contour = find_document(frame)

    if document_contour is not None:
        # Apply perspective transformation to get a top-down view
        warped = four_point_transform(frame, document_contour.reshape(4, 2))

        # Display the original and transformed frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Transformed Frame', warped)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

67. **Real-Time Face Swapping with Dlib and OpenCV:**

```python
# Install the `opencv-python` and `dlib` libraries (if not installed): pip install opencv-python dlib

import cv2
import dlib
import numpy as np

# Load the pre-trained facial landmark predictor
predictor_path = 'path/to/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Load the images for face swapping
face_image_path = 'path/to/face_image.jpg'
face_image = cv2.imread(face_image_path)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Extract the face region from the image
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        face_region = frame[y:y + h, x:x + w]

        # Resize the face image to match the size of the face region
        resized_face_image = cv2.resize(face_image, (w, h))

        # Create a mask for the face region
        mask = np.zeros_like(face_region, dtype=np.uint8)

        # Extract the facial landmarks and create a mask for the face
        points = np.array([(landmark.x, landmark.y) for landmark in landmarks.parts()])
        convex_hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convex_hull, (255, 255, 255))

        # Apply the mask to the face region and the resized face image
        face_region = cv2.bitwise_and(face_region, mask)
        resized_face_image = cv2.bitwise_and(resized_face_image, mask)

        # Combine the face region and the resized face image
        result = cv2.add(face_region, resized_face_image)

        # Replace the face region in the original frame with the result
        frame[y:y + h, x:x + w] = result

    # Display the real-time face swapping result
    cv2.imshow('Face Swapping', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover real-time document scanning using OpenCV and face swapping with Dlib and OpenCV. Feel free to experiment with these examples and adapt them to your specific projects!





68. **Real-Time Text Detection with EAST and OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Load the EAST text detector model
net = cv2.dnn.readNet('path/to/east_model.pb')

def text_detection(image):
    # Resize the image to have a width divisible by 32
    (H, W) = image.shape[:2]
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # Resize the image and get the new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Construct a blob from the image and perform a forward pass
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

    # Decode the predictions and apply non-maxima suppression
    rects, confidences = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Rescale the bounding boxes back to the original image size
    boxes = rescale_boxes(boxes, rW, rH)

    # Draw bounding boxes on the image
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)

    return image

def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def rescale_boxes(boxes, rW, rH):
    boxes[:, 0] = np.round(boxes[:, 0] * rW)
    boxes[:, 1] = np.round(boxes[:, 1] * rH)
    boxes[:, 2] = np.round(boxes[:, 2] * rW)
    boxes[:, 3] = np.round(boxes[:, 3] * rH)
    return boxes

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Perform text detection on the frame
    result_frame = text_detection(frame)

    # Display the real-time text detection result
    cv2.imshow('Text Detection', result_frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

69. **Real-Time Human Pose Estimation with OpenPose:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Load the OpenPose model
net = cv2.dnn.readNetFromTensorflow('path/to/openpose_model.pb')

def pose_estimation(image):
    # Prepare the image for OpenPose
    frameWidth = image.shape[1]
    frameHeight = image.shape[0]
    net.setInput(cv2.dnn.blobFromImage(image, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))

    # Run forward pass through the network
    output = net.forward()

    # Extract key points for body parts
    points = []
    for i in range(18):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        if prob > 0.1:
            x = int((frameWidth * point[0]) / output.shape[3])
            y = int((frameHeight * point[1]) / output.shape[2])
            points.append((x, y))
        else:
            points.append(None)

   

return points

def draw_pose(image, points):
    # Draw lines connecting body parts
    for pair in [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]:
        if points[pair[0]] and points[pair[1]]:
            cv2.line(image, points[pair[0]], points[pair[1]], (0, 255, 0), 2)

    # Draw keypoints
    for point in points:
        if point:
            cv2.circle(image, point, 5, (0, 0, 255), -1)

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Perform pose estimation on the frame
    key_points = pose_estimation(frame)

    # Draw the pose on the frame
    draw_pose(frame, key_points)

    # Display the real-time pose estimation result
    cv2.imshow('Pose Estimation', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover real-time text detection with EAST and OpenCV, as well as real-time human pose estimation with OpenPose. Feel free to experiment with these examples and adapt them to your specific projects!





70. **Real-Time Facial Expression Recognition with Deep Learning:**

```python
# Install the `opencv-python`, `tensorflow`, and `keras` libraries (if not installed): pip install opencv-python tensorflow keras

import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained facial expression recognition model
model_path = 'path/to/facial_expression_model.h5'
emotion_model = load_model(model_path, compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    emotion_probabilities = emotion_model.predict(face_image)
    emotion_label = emotion_labels[np.argmax(emotion_probabilities)]
    return emotion_label

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = frame[y:y + h, x:x + w]

        # Perform emotion prediction
        emotion = predict_emotion(face_image)

        # Display the emotion label
        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time facial expression recognition result
    cv2.imshow('Facial Expression Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

71. **Real-Time Age and Gender Estimation with Deep Learning:**

```python
# Install the `opencv-python`, `tensorflow`, and `keras` libraries (if not installed): pip install opencv-python tensorflow keras

import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained age and gender estimation model
age_gender_model_path = 'path/to/age_gender_model.h5'
age_gender_model = load_model(age_gender_model_path, compile=False)

def predict_age_gender(face_image):
    # Preprocess the face image
    face_image = cv2.resize(face_image, (48, 48))
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 3])
    face_image = face_image / 255.0

    # Perform age and gender prediction
    predictions = age_gender_model.predict(face_image)
    age_label = int(np.round(predictions[0][0] * 100))
    gender_label = 'Male' if predictions[1][0] < 0.5 else 'Female'

    return age_label, gender_label

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = frame[y:y + h, x:x + w]

        # Perform age and gender prediction
        age, gender = predict_age_gender(face_image)

        # Display the age and gender labels
        label = f'Age: {age} | Gender: {gender}'
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time age and gender estimation result
    cv2.imshow('Age and Gender Estimation', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover real-time facial expression recognition, age, and gender estimation using deep learning models. Feel free to experiment with these examples and adapt them to your specific projects!





72. **Real-Time Hand Gesture Recognition with MediaPipe:**

```python
# Install the `mediapipe` library (if not installed): pip install mediapipe

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB for processing with MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in hand_landmarks.landmark]

            # Draw hand landmarks on the frame
            for landmark in landmarks:
                cv2.circle(frame, landmark, 5, (0, 255, 0), -1)

            # Get the gesture based on hand landmarks (customize as needed)
            gesture = get_gesture(landmarks)
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the real-time hand gesture recognition result
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

For the `get_gesture` function, you can implement your logic to determine the gesture based on the hand landmarks. You might need to experiment and customize this part depending on your specific use case.

73. **Real-Time Object Tracking with OpenCV and CSRT Tracker:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Create a video capture object
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame = cap.read()

# Select a region of interest (ROI) for tracking
roi = cv2.selectROI(frame, fromCenter=False)

# Initialize the CSRT tracker
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, roi)

while True:
    # Read a new frame
    ret, frame = cap.read()

    # Update the tracker
    success, roi = tracker.update(frame)

    # Draw the tracking rectangle on the frame
    if success:
        (x, y, w, h) = tuple(map(int, roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time object tracking result
    cv2.imshow('Object Tracking', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

In this example, the CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) tracker is used for object tracking. You can experiment with other trackers provided by OpenCV, such as KCF (Kernelized Correlation Filters) or MOSSE (Minimum Output Sum of Squared Error).

Feel free to experiment with and customize these examples based on your specific needs. These cover hand gesture recognition with MediaPipe and real-time object tracking with OpenCV's CSRT tracker.





74. **Real-Time Lane Detection in a Video Stream:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest (ROI)
    height, width = edges.shape
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    # Apply Hough transform to detect lines
    lines = hough_lines(roi, 1, np.pi / 180, 50, 100, 160)

    # Combine the original frame with the detected lines
    result = cv2.addWeighted(frame, 0.8, lines, 1, 0)

    # Display the real-time lane detection result
    cv2.imshow('Lane Detection', result)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

75. **Real-Time Background Subtraction with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Create a video capture object
cap = cv2.VideoCapture(0)

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Display the real-time background subtraction result
    cv2.imshow('Background Subtraction', fg_mask)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time lane detection in a video stream and real-time background subtraction with OpenCV.





76. **Real-Time Color Detection in a Video Stream:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

def color_detection(frame, lower_bound, upper_bound):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Define the lower and upper bounds for color detection (adjust as needed)
    lower_bound = np.array([30, 50, 50])
    upper_bound = np.array([60, 255, 255])

    # Apply color detection
    color_detected = color_detection(frame, lower_bound, upper_bound)

    # Display the real-time color detection result
    cv2.imshow('Color Detection', color_detected)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

77. **Real-Time Face Recognition with OpenCV and Haar Cascades:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained Haar cascade for eye detection (optional)
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Optionally, detect eyes within the detected face region
        # roi_gray = gray[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

    # Display the real-time face recognition result
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time color detection in a video stream and real-time face recognition with OpenCV and Haar cascades.





78. **Real-Time QR Code Detection and Decoding with OpenCV:**

```python
# Install the `opencv-python` and `pyzbar` libraries (if not installed): pip install opencv-python pyzbar

import cv2
from pyzbar.pyzbar import decode

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect and decode QR codes in the frame
    qr_codes = decode(gray)

    for qr_code in qr_codes:
        # Extract QR code information
        qr_data = qr_code.data.decode('utf-8')
        qr_type = qr_code.type

        # Draw a rectangle around the QR code
        rect_points = qr_code.polygon
        if rect_points:
            pts = np.array(rect_points, dtype=int)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display QR code information
        cv2.putText(frame, f'Type: {qr_type}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Data: {qr_data}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the real-time QR code detection result
    cv2.imshow('QR Code Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

79. **Real-Time Emotion Detection with OpenCV and Deep Learning:**

```python
# Install the `opencv-python`, `tensorflow`, and `keras` libraries (if not installed): pip install opencv-python tensorflow keras

import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained emotion detection model
emotion_model_path = 'path/to/emotion_model.h5'
emotion_model = load_model(emotion_model_path, compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    face_image = face_image / 255.0

    emotion_probabilities = emotion_model.predict(face_image)
    emotion_label = emotion_labels[np.argmax(emotion_probabilities)]
    return emotion_label

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = frame[y:y + h, x:x + w]

        # Perform emotion prediction
        emotion = predict_emotion(face_image)

        # Display the emotion label
        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time emotion detection result
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time QR code detection and decoding, as well as real-time emotion detection with OpenCV and deep learning.





80. **Real-Time Handwritten Digit Recognition with OpenCV and MNIST Model:**

```python
# Install the `opencv-python`, `tensorflow`, and `keras` libraries (if not installed): pip install opencv-python tensorflow keras

import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained MNIST digit recognition model
mnist_model_path = 'path/to/mnist_model.h5'
mnist_model = load_model(mnist_model_path)

def predict_digit(image):
    # Preprocess the image for MNIST model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = np.reshape(image, [1, 28, 28, 1])
    image = image / 255.0

    # Predict the digit
    digit_probabilities = mnist_model.predict(image)
    digit_label = np.argmax(digit_probabilities)
    return digit_label

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define a region of interest (ROI) for digit recognition
    roi = gray_frame[100:300, 100:300]

    # Predict the digit in the ROI
    digit_label = predict_digit(roi)

    # Display the digit label
    cv2.putText(frame, f'Digit: {digit_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    # Display the real-time handwritten digit recognition result
    cv2.imshow('Digit Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

81. **Real-Time Object Detection with YOLO (You Only Look Once):**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Load the pre-trained YOLO model
yolo_net = cv2.dnn.readNet('path/to/yolov3.weights', 'path/to/yolov3.cfg')

# Load the COCO names file (contains class names)
with open('path/to/coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Convert the frame to a blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLO model
    yolo_net.setInput(blob)

    # Get the output layer names
    output_layer_names = yolo_net.getUnconnectedOutLayersNames()

    # Run forward pass through the network
    detections = yolo_net.forward(output_layer_names)

    # Process the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions
            if confidence > 0.5:
                # Scale the bounding box coordinates back to the original frame size
                box = obj[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")

                # Calculate the top-left corner of the bounding box
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + int(box_width), y + int(box_height)), (0, 255, 0), 2)
                label = f'{classes[class_id]}: {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the real-time object detection result
    cv2.imshow('Object Detection (YOLO)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

These examples cover real-time handwritten digit recognition using the MNIST model, and real-time object detection using the YOLO (You Only Look Once) model. Feel free to experiment with and customize these examples based on your specific needs.





82. **Real-Time Text Recognition with Tesseract OCR:**

```python
# Install the `opencv-python` and `pytesseract` libraries (if not installed): pip install opencv-python pytesseract

import cv2
import pytesseract

# Set the path to the Tesseract OCR executable (update as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance text visibility (adjust parameters as needed)
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Perform text recognition using Tesseract OCR
    text = pytesseract.image_to_string(thresholded)

    # Display the recognized text
    cv2.putText(frame, f'Text: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the real-time text recognition result
    cv2.imshow('Text Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

83. **Real-Time Barcode Scanning with ZBar and OpenCV:**

```python
# Install the `opencv-python` and `zbar` libraries (if not installed): pip install opencv-python zbar

import cv2
from pyzbar.pyzbar import decode

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Decode barcodes in the frame
    barcodes = decode(gray)

    for barcode in barcodes:
        # Extract barcode information
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type

        # Draw a rectangle around the barcode
        rect_points = barcode.polygon
        if rect_points:
            pts = np.array(rect_points, dtype=int)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display barcode information
        cv2.putText(frame, f'Type: {barcode_type}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Data: {barcode_data}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the real-time barcode scanning result
    cv2.imshow('Barcode Scanning', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time text recognition using Tesseract OCR and real-time barcode scanning with ZBar and OpenCV.




84. **Real-Time Document Scanner with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the contour with the maximum area
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Draw the contour and its convex hull on the frame
    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)

    # Perform four-point perspective transform on the original frame
    warped = four_point_transform(frame, approx.reshape(4, 2))

    # Display the real-time document scanning result
    cv2.imshow('Document Scanner', warped)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

85. **Real-Time Image Segmentation with GrabCut in OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Initialize the foreground and background models for GrabCut
rect = (50, 50, 400, 300)

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Create a mask and initialize the foreground and background models
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm to segment the image
    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to create a binary mask for the foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply the frame with the binary mask to obtain the segmented image
    result = frame * mask2[:, :, np.newaxis]

    # Display the real-time image segmentation result
    cv2.imshow('Image Segmentation', result)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time document scanning with OpenCV and real-time image segmentation using the GrabCut algorithm.





86. **Real-Time Style Transfer with OpenCV and Neural Style Transfer:**

```python
# Install the `opencv-python` and `neural-style` libraries (if not installed): pip install opencv-python neural-style

import cv2
import subprocess
import numpy as np

# Define the path to the neural-style executable
neural_style_path = 'path/to/neural-style'

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Save the frame as a temporary image file
    cv2.imwrite('temp_frame.jpg', frame)

    # Define the path to the temporary image file
    input_image_path = 'temp_frame.jpg'

    # Define the path to the style image for neural style transfer
    style_image_path = 'path/to/style_image.jpg'

    # Define the output image path
    output_image_path = 'output_frame.jpg'

    # Perform neural style transfer using the neural-style executable
    subprocess.run([neural_style_path, '--content', input_image_path, '--styles', style_image_path, '--output', output_image_path])

    # Read the stylized output image
    stylized_frame = cv2.imread(output_image_path)

    # Display the real-time stylized frame
    cv2.imshow('Neural Style Transfer', stylized_frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Remove the temporary image file
subprocess.run(['rm', 'temp_frame.jpg'])

cap.release()
cv2.destroyAllWindows()
```

Note: Ensure that you have the `neural-style` executable installed and available in your system's PATH. You can find the neural-style repository on GitHub (https://github.com/jcjohnson/neural-style).

87. **Real-Time Hand Tracking with OpenCV and MediaPipe:**

```python
# Install the `opencv-python` and `mediapipe` libraries (if not installed): pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB for processing with MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in hand_landmarks.landmark]

            # Draw hand landmarks and connect them with lines
            for i, landmark in enumerate(landmarks):
                cv2.circle(frame, landmark, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(frame, landmarks[i - 1], landmark, (0, 255, 0), 2)

    # Display the real-time hand tracking result
    cv2.imshow('Hand Tracking', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Ensure that you have the `mediapipe` library installed.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time style transfer using Neural Style Transfer and real-time hand tracking using MediaPipe.




88. **Real-Time Object Tracking with OpenCV and MOSSE Tracker:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Create a video capture object
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame = cap.read()

# Select the ROI (Region of Interest) for tracking
bbox = cv2.selectROI(frame, False)

# Initialize the MOSSE tracker
tracker = cv2.TrackerMOSSE_create()

# Initialize the tracker with the selected ROI
tracker.init(frame, bbox)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Update the tracker with the current frame
    ret, bbox = tracker.update(frame)

    # Convert the bounding box coordinates to integers
    bbox = tuple(map(int, bbox))

    # Draw the bounding box on the frame
    cv2.rectangle(frame, bbox, (0, 255, 0), 2)

    # Display the real-time object tracking result
    cv2.imshow('Object Tracking (MOSSE)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

89. **Real-Time Facial Landmark Detection with OpenCV and Dlib:**

```python
# Install the `opencv-python` and `dlib` libraries (if not installed): pip install opencv-python dlib

import cv2
import dlib

# Initialize the Dlib facial landmark detector
predictor_path = 'path/to/shape_predictor_68_face_landmarks.dat'  # Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for facial landmark detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray_frame)

    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray_frame, face)

        # Draw facial landmarks on the frame
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display the real-time facial landmark detection result
    cv2.imshow('Facial Landmark Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the pre-trained facial landmark predictor from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and update the `predictor_path` accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time object tracking using the MOSSE tracker, and real-time facial landmark detection using Dlib.





90. **Real-Time Human Pose Estimation with OpenCV and PoseNet:**

```python
# Install the `opencv-python` and `posenet` libraries (if not installed): pip install opencv-python posenet

import cv2
from posenet import PoseNet, draw_skel_and_kp

# Initialize PoseNet
net = PoseNet()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Perform human pose estimation using PoseNet
    keypoints, _ = net.process_frame(frame)

    # Draw skeleton and keypoints on the frame
    draw_skel_and_kp(frame, keypoints)

    # Display the real-time human pose estimation result
    cv2.imshow('Human Pose Estimation (PoseNet)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Install the `posenet` library using `pip install posenet`. Additionally, ensure that you have TensorFlow installed.

91. **Real-Time Image Stitching with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Create a video capture object
cap = cv2.VideoCapture(0)

# Read the first frame
_, frame1 = cap.read()

# Read the second frame
_, frame2 = cap.read()

while True:
    # Perform feature matching using ORB
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    # Use the BFMatcher to find the best matches between the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches on the frames
    img_matches = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the real-time image stitching result
    cv2.imshow('Image Stitching', img_matches)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time human pose estimation using PoseNet and real-time image stitching with OpenCV.




92. **Real-Time Gesture Recognition with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np
from sklearn.metrics import pairwise

# Create a video capture object
cap = cv2.VideoCapture(0)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# ROI coordinates (adjust based on your setup)
top, right, bottom, left = 100, 300, 300, 600

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Apply background subtraction
    roi = frame[top:bottom, right:left]
    fg_mask = bg_subtractor.apply(roi)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the contour with the maximum area
        max_contour = max(contours, key=cv2.contourArea)

        # Calculate the convex hull of the hand contour
        hull = cv2.convexHull(max_contour)

        # Draw the hand contour and convex hull on the frame
        cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(roi, [hull], -1, (0, 0, 255), 2)

        # Calculate the defects in the convex hull
        defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Calculate the cosine of the angle between the fingers
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                # Ignore angles close to 180 degrees
                if angle <= np.pi / 2:
                    cv2.circle(roi, far, 5, (0, 0, 255), -1)

        # Count the number of fingers (based on defects)
        finger_count = np.sum(defects[:, 0, 3] > 1000)

        # Display the finger count
        cv2.putText(frame, f'Finger Count: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the real-time gesture recognition result
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

93. **Real-Time Face Recognition with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `face_recognition` libraries (if not installed): pip install opencv-python dlib face_recognition

import cv2
import face_recognition

# Load known face image and encode it
known_image = face_recognition.load_image_file('path/to/known_face.jpg')
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Create arrays of known face encodings and corresponding labels
known_face_encodings = [known_face_encoding]
known_face_labels = ['Known Face']

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = 'Unknown'

        # If a match is found, use the label of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_labels[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the real-time face recognition result
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Update the paths and filenames in the face recognition example with your own known face image.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time gesture recognition with OpenCV and real-time face recognition with Dlib and face_recognition library.




94. **Real-Time Emotion Detection with OpenCV and Deep Learning:**

```python
# Install the `opencv-python`, `tensorflow`, and `keras` libraries (if not installed): pip install opencv-python tensorflow keras

import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained emotion detection model
emotion_model_path = 'path/to/emotion_model.h5'  # Download from https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/emotion_detector_models/emotion_detector_model.h5
emotion_model = load_model(emotion_model_path)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the input size of the model
    input_frame = cv2.resize(gray_frame, (48, 48))

    # Reshape the input frame
    input_frame = np.reshape(input_frame, (1, 48, 48, 1))

    # Normalize the input frame
    input_frame = input_frame / 255.0

    # Predict the emotion
    emotion_probabilities = emotion_model.predict(input_frame)
    emotion_label = emotion_labels[np.argmax(emotion_probabilities)]

    # Display the predicted emotion
    cv2.putText(frame, f'Emotion: {emotion_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the real-time emotion detection result
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the pre-trained emotion detection model from [here](https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/emotion_detector_models/emotion_detector_model.h5) and update the `emotion_model_path` accordingly.

95. **Real-Time Hand Gesture Recognition with OpenCV and CNN:**

```python
# Install the `opencv-python`, `tensorflow`, and `keras` libraries (if not installed): pip install opencv-python tensorflow keras

import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained hand gesture recognition model
gesture_model_path = 'path/to/gesture_model.h5'  # Create or download a hand gesture recognition model
gesture_model = load_model(gesture_model_path)

# Define gesture labels
gesture_labels = ['Fist', 'Five', 'Point', 'Swing', 'Ok']

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Resize the frame to match the input size of the model
    input_frame = cv2.resize(frame, (64, 64))

    # Reshape the input frame
    input_frame = np.reshape(input_frame, (1, 64, 64, 3))

    # Normalize the input frame
    input_frame = input_frame / 255.0

    # Predict the hand gesture
    gesture_probabilities = gesture_model.predict(input_frame)
    gesture_label = gesture_labels[np.argmax(gesture_probabilities)]

    # Display the predicted hand gesture
    cv2.putText(frame, f'Gesture: {gesture_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the real-time hand gesture recognition result
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Create or download a hand gesture recognition model and update the `gesture_model_path` accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time emotion detection with OpenCV and deep learning, and real-time hand gesture recognition with OpenCV and CNN.




96. **Real-Time Object Detection with OpenCV and YOLO:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Load the YOLO model and configuration files
yolo_net = cv2.dnn.readNet('path/to/yolov3.weights', 'path/to/yolov3.cfg')
yolo_classes = open('path/to/coco.names').read().strip().split('\n')
yolo_colors = np.random.randint(0, 255, size=(len(yolo_classes), 3), dtype='uint8')

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLO network
    yolo_net.setInput(blob)

    # Get the output layer names
    output_layer_names = yolo_net.getUnconnectedOutLayersNames()

    # Run forward pass to get predictions
    detections = yolo_net.forward(output_layer_names)

    # Process each detection
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions
            if confidence > 0.5:
                # Scale bounding box coordinates to the original frame size
                box = obj[0:4] * np.array([width, height, width, height])
                (x, y, w, h) = box.astype('int')

                # Draw bounding box and label on the frame
                color = [int(c) for c in yolo_colors[class_id]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f'{yolo_classes[class_id]}: {confidence:.2f}'
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the real-time object detection result
    cv2.imshow('Object Detection (YOLO)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the YOLOv3 weights file, configuration file, and COCO names file from the official YOLO website (https://pjreddie.com/darknet/yolo/) and update the paths accordingly.

97. **Real-Time Scene Recognition with OpenCV and MobileNet:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Load the MobileNet model and labels
mobilenet_net = cv2.dnn.readNetFromTensorflow('path/to/mobilenet_v2/frozen_inference_graph.pb', 'path/to/mobilenet_v2/labelmap.pbtxt')
mobilenet_classes = open('path/to/mobilenet_v2/labels.txt').read().strip().split('\n')
mobilenet_colors = np.random.randint(0, 255, size=(len(mobilenet_classes), 3), dtype='uint8')

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (300, 300), swapRB=True, crop=False)

    # Set the input to the MobileNet network
    mobilenet_net.setInput(blob)

    # Get the output layer names
    output_layer_names = ['num_detections', 'detection_classes', 'detection_scores', 'detection_boxes']

    # Run forward pass to get predictions
    detections = mobilenet_net.forward(output_layer_names)

    # Process each detection
    for i in range(int(detections['num_detections'][0])):
        class_id = int(detections['detection_classes'][0][i])
        score = detections['detection_scores'][0][i]

        # Filter out weak predictions
        if score > 0.5:
            # Scale bounding box coordinates to the original frame size
            box = detections['detection_boxes'][0][i] * np.array([width, height, width, height])
            (x, y, w, h) = box.astype('int')

            # Draw bounding box and label on the frame
            color = [int(c) for c in mobilenet_colors[class_id]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f'{mobilenet_classes[class_id]}: {score:.2f}'
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the real-time scene recognition result
    cv2.imshow('Scene Recognition (MobileNet)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the MobileNet SSD model files and labels

from the TensorFlow GitHub repository (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and update the paths accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time object detection with YOLO and real-time scene recognition with MobileNet.




98. **Real-Time Color Detection with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color to detect (here, green)
    lower_bound = np.array([40, 40, 40])
    upper_bound = np.array([80, 255, 255])

    # Create a mask for the specified color range
    color_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Apply the mask to the original frame
    color_result = cv2.bitwise_and(frame, frame, mask=color_mask)

    # Display the real-time color detection result
    cv2.imshow('Color Detection', color_result)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

99. **Real-Time Lane Detection with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Define region of interest (ROI) for lane detection
    height, width = frame.shape[:2]
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, [np.array(roi_vertices, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)

    # Perform Hough line detection on the masked edges
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)

    # Draw detected lines on the frame
    line_image = np.zeros_like(frame)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Combine the original frame with the detected lines
    lane_detection_result = cv2.addWeighted(frame, 1, line_image, 1, 0)

    # Display the real-time lane detection result
    cv2.imshow('Lane Detection', lane_detection_result)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time color detection, and real-time lane detection with OpenCV.





100. **Real-Time Vehicle Detection with OpenCV and Haarcascades:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Load the pre-trained Haarcascades classifier for vehicle detection
vehicle_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles in the frame
    vehicles = vehicle_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time vehicle detection result
    cv2.imshow('Vehicle Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Ensure that you have the `haarcascade_car.xml` file available. You can find it in the OpenCV GitHub repository (https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_car.xml).

101. **Real-Time Text Detection with OpenCV and EAST Text Detector:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Load the pre-trained EAST text detector model
east_net = cv2.dnn.readNet('path/to/east_text_detector.pb')

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Set the input to the EAST text detector
    east_net.setInput(blob)

    # Get the output layer names
    output_layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    # Run forward pass to get predictions
    scores, geometry = east_net.forward(output_layer_names)

    # Post-process the predictions
    rectangles, confidences = cv2.dnn.NMSBoxesRotated(
        boxes=np.array([[[0, 0, width, height]]]),
        scores=scores,
        score_threshold=0.5,
        nms_threshold=0.4
    )

    # Draw rectangles around detected text
    for i in range(len(rectangles)):
        box = rectangles[i]
        confidence = confidences[i]

        if confidence > 0.5:
            # Extract the rotated rectangle parameters
            (cx, cy), (w, h), angle = box
            box_pts = cv2.boxPoints(((cx, cy), (w, h), angle))

            # Convert the box coordinates to integers
            box_pts = np.int0(box_pts)

            # Draw the rotated rectangle on the frame
            cv2.drawContours(frame, [box_pts], 0, (0, 255, 0), 2)

    # Display the real-time text detection result
    cv2.imshow('Text Detection (EAST)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the EAST text detector model file (`east_text_detector.pb`) from the official EAST GitHub repository (https://github.com/argman/EAST) and update the path accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time vehicle detection with Haarcascades and real-time text detection with the EAST Text Detector.





102. **Real-Time Barcode and QR Code Detection with OpenCV:**

```python
# Install the `opencv-python` and `pyzbar` libraries (if not installed): pip install opencv-python pyzbar

import cv2
from pyzbar.pyzbar import decode

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Decode barcodes and QR codes in the frame
    decoded_objects = decode(frame)

    # Loop through the detected objects
    for obj in decoded_objects:
        # Extract the barcode or QR code data
        barcode_data = obj.data.decode('utf-8')

        # Draw a rectangle around the detected object
        points = obj.polygon
        if len(points) == 4:
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Display the data of the detected barcode or QR code
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, barcode_data, (points[0][0], points[0][1] - 10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the real-time barcode and QR code detection result
    cv2.imshow('Barcode and QR Code Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

103. **Real-Time Handwriting Recognition with OpenCV and Tesseract OCR:**

```python
# Install the `opencv-python` and `pytesseract` libraries (if not installed): pip install opencv-python pytesseract

import cv2
import pytesseract

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path accordingly

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance text visibility
    _, thresholded_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY_INV)

    # Use Tesseract OCR to perform handwriting recognition
    text = pytesseract.image_to_string(thresholded_frame)

    # Display the real-time handwriting recognition result
    cv2.putText(frame, f'Text: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Handwriting Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Ensure that Tesseract OCR is installed on your system, and update the path to the Tesseract OCR executable accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time barcode and QR code detection with OpenCV, and real-time handwriting recognition with Tesseract OCR.





104. **Real-Time Document Scanner with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

# Initialize the document scanner
scanner = cv2.QRCodeDetector()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Find contours in the edged frame
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to find the document contour
    document_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            document_contour = contour
            break

    # If a document contour is found, apply perspective transformation
    if document_contour is not None:
        # Calculate the perimeter of the document contour
        perimeter = cv2.arcLength(document_contour, True)

        # Approximate the polygonal curve to get the four corners of the document
        document_corners = cv2.approxPolyDP(document_contour, 0.02 * perimeter, True)

        # Draw the document contour on the frame
        cv2.drawContours(frame, [document_corners], -1, (0, 255, 0), 2)

        # Apply perspective transformation to get a top-down view of the document
        document_transformed = scanner.getTransform(frame, document_corners)
        document_transformed = cv2.cvtColor(document_transformed, cv2.COLOR_BGR2GRAY)

        # Display the real-time document scanner result
        cv2.imshow('Document Scanner', document_transformed)

    else:
        # Display the original frame if no document contour is found
        cv2.imshow('Document Scanner', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

105. **Real-Time Face Swapping with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `face_recognition` libraries (if not installed): pip install opencv-python dlib face_recognition

import cv2
import dlib
import face_recognition

# Load the source and target face images
source_image = face_recognition.load_image_file('path/to/source_face.jpg')  # Replace with the path to the source face image
target_image = face_recognition.load_image_file('path/to/target_face.jpg')  # Replace with the path to the target face image

# Find face landmarks in the source and target images
source_face_landmarks = face_recognition.face_landmarks(source_image)[0]
target_face_landmarks = face_recognition.face_landmarks(target_image)[0]

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Load the face detector from dlib
face_detector = dlib.get_frontal_face_detector()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame using the dlib face detector
    faces = face_detector(frame)

    # Check if faces are found
    if len(faces) > 0:
        # Extract face landmarks for the first face found
        landmarks = face_recognition.face_landmarks(rgb_frame, faces=[faces[0]])

        # Check if landmarks are found
        if len(landmarks) > 0:
            # Extract the landmarks for the first face found
            face_landmarks = landmarks[0]

            # Swap the faces in the frame
            face_swapped = face_recognition.face_swap(frame, source_face_landmarks, face_landmarks)

            # Display the real-time face swapping result
            cv2.imshow('Face Swapping', face_swapped)

        else:
            # Display the original frame if landmarks are not found
            cv2.imshow('Face Swapping', frame)
    else:
        # Display the original frame if no faces are found
        cv2.imshow('Face Swapping', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Update the paths to the source and target face images in the face swapping example.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time document scanning with OpenCV and face swapping with OpenCV and Dlib.




106. **Real-Time Facial Landmark Detection and Filters with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `face_recognition` libraries (if not installed): pip install opencv-python dlib face_recognition

import cv2
import dlib
import face_recognition
import numpy as np

# Load a pre-trained facial landmark predictor model from dlib
predictor_path = 'path/to/shape_predictor_68_face_landmarks.dat'  # Replace with the path to the model file
face_predictor = dlib.shape_predictor(predictor_path)

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_predictor(gray_frame)

    # Loop through the detected faces
    for face in faces:
        # Extract facial landmarks
        landmarks = face_recognition.face_landmarks(frame, [face])

        # Draw facial landmarks on the frame
        for landmark_type, landmarks_list in landmarks[0].items():
            for landmark in landmarks_list:
                cv2.circle(frame, landmark, 1, (0, 255, 0), -1)

        # Apply a filter (e.g., sunglasses) to the detected face
        filter_image = cv2.imread('path/to/sunglasses.png')  # Replace with the path to the filter image
        filter_image = cv2.resize(filter_image, (face.width, face.height))

        # Calculate the position for placing the filter
        x, y = landmarks[0]['nose_bridge'][3]
        x -= face.width // 2
        y -= face.height // 2

        # Overlay the filter on the frame
        roi = frame[y:y + filter_image.shape[0], x:x + filter_image.shape[1]]
        alpha_channel = filter_image[:, :, 3] / 255.0
        for c in range(0, 3):
            roi[:, :, c] = (1.0 - alpha_channel) * roi[:, :, c] + alpha_channel * filter_image[:, :, c]

    # Display the real-time facial landmark detection and filter result
    cv2.imshow('Facial Landmark Detection and Filters', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the pre-trained facial landmark predictor model (`shape_predictor_68_face_landmarks.dat`) from the dlib website (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and update the path accordingly. Also, replace the path to the filter image with the path to your own filter image.

107. **Real-Time Virtual Makeup Application with OpenCV:**

```python
# Install the `opencv-python`, `dlib`, and `face_recognition` libraries (if not installed): pip install opencv-python dlib face_recognition

import cv2
import dlib
import face_recognition
import numpy as np

# Load a pre-trained facial landmark predictor model from dlib
predictor_path = 'path/to/shape_predictor_68_face_landmarks.dat'  # Replace with the path to the model file
face_predictor = dlib.shape_predictor(predictor_path)

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Load virtual makeup images (e.g., lipstick and eyeshadow)
lipstick_image = cv2.imread('path/to/lipstick.png')  # Replace with the path to the lipstick image
eyeshadow_image = cv2.imread('path/to/eyeshadow.png')  # Replace with the path to the eyeshadow image

# Resize virtual makeup images
lipstick_image = cv2.resize(lipstick_image, (100, 50))
eyeshadow_image = cv2.resize(eyeshadow_image, (150, 100))

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_predictor(gray_frame)

    # Loop through the detected faces
    for face in faces:
        # Extract facial landmarks
        landmarks = face_recognition.face_landmarks(frame, [face])

        # Apply lipstick to the detected face
        lips_top = landmarks[0]['top_lip'][0]
        lips_bottom = landmarks[0]['bottom_lip'][-1]
        x, y = lips_top
        x -= lipstick_image.shape[1] // 2
        y -= lipstick_image.shape[0] // 2
        roi = frame[y:y + lipstick_image.shape[0], x:x + lipstick_image.shape[1]]
        alpha_channel = lipstick_image[:, :, 3] / 255.0
        for c in range(0, 3):
            roi[:, :, c] = (1.0 - alpha_channel) * roi[:, :, c] + alpha_channel * lipstick_image[:, :, c]

        # Apply eyeshadow to the detected face
        eyeshadow_left = landmarks[0]['left_eye'][0]
        eyeshadow_right = landmarks[0]['right_eye'][-1]
        x, y = eyeshadow_left
        x -= eyeshadow_image.shape[1] // 2
        y -= eyeshadow_image.shape[0] // 2
        roi = frame[y:y + eyeshadow_image.shape[0], x:x + eyeshadow_image.shape[1]]
        alpha_channel = eyeshadow_image[:, :, 3] / 255.0
        for c in range(0, 3):
            roi[:, :, c] = (1.0 - alpha_channel) * roi[:, :, c] + alpha_channel * eyeshadow_image[:, :, c]

    # Display the real-time virtual makeup application result
    cv2.imshow('Virtual Makeup Application', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the virtual makeup images (e.g., lipstick and eyeshadow) and replace the paths accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time facial landmark detection and filters, real-time virtual makeup application with OpenCV.




108. **Real-Time Emotion Recognition with OpenCV and Deep Learning:**

```python
# Install the `opencv-python` and `tensorflow` libraries (if not installed): pip install opencv-python tensorflow

import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained emotion recognition model
emotion_model = tf.keras.models.load_model('path/to/emotion_model.h5')  # Replace with the path to the model file

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the input size of the emotion model
    resized_frame = cv2.resize(gray_frame, (48, 48))

    # Normalize the pixel values to be in the range [0, 1]
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to create a batch size of 1
    input_data = np.expand_dims(np.expand_dims(normalized_frame, -1), 0)

    # Predict the emotion label for the current frame
    emotion_prediction = emotion_model.predict(input_data)[0]
    predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]

    # Display the emotion label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), font, 0.8, (0, 255, 0), 2)

    # Display the real-time emotion recognition result
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the pre-trained emotion recognition model file (`emotion_model.h5`) and update the path accordingly.

109. **Real-Time Gesture Recognition with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Apply background subtraction to detect hand gestures
    fg_mask = bg_subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    # Draw bounding boxes around valid contours
    for contour in valid_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time gesture recognition result
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

110. **Real-Time Sign Language Recognition with OpenCV and TensorFlow:**

```python
# Install the `opencv-python` and `tensorflow` libraries (if not installed): pip install opencv-python tensorflow

import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained sign language recognition model
sign_language_model = tf.keras.models.load_model('path/to/sign_language_model.h5')  # Replace with the path to the model file

# Define sign language labels
sign_language_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SPACE', 'DELETE']

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the input size of the sign language model
    resized_frame = cv2.resize(gray_frame, (64, 64

))

    # Normalize the pixel values to be in the range [0, 1]
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to create a batch size of 1
    input_data = np.expand_dims(np.expand_dims(normalized_frame, -1), 0)

    # Predict the sign language label for the current frame
    sign_language_prediction = sign_language_model.predict(input_data)[0]
    predicted_sign_language = sign_language_labels[np.argmax(sign_language_prediction)]

    # Display the predicted sign language label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Prediction: {predicted_sign_language}', (10, 30), font, 0.8, (0, 255, 0), 2)

    # Display the real-time sign language recognition result
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the pre-trained sign language recognition model file (`sign_language_model.h5`) and update the path accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time emotion recognition, gesture recognition, and sign language recognition with OpenCV and deep learning.




111. **Real-Time Object Tracking with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

# Create the object tracker (here, using the KCF tracker)
tracker = cv2.TrackerKCF_create()

# Read the first frame
ret, frame = cap.read()

# Select a bounding box around the object to be tracked
bbox = cv2.selectROI(frame, False)

# Initialize the tracker with the bounding box
tracker.init(frame, bbox)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Update the tracker
    success, bbox = tracker.update(frame)

    # Draw the bounding box on the frame
    if success:
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time object tracking result
    cv2.imshow('Object Tracking', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

112. **Real-Time People Counting with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize people counter
people_count = 0

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Apply background subtraction to detect people
    fg_mask = bg_subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    # Update people count based on the number of valid contours
    people_count = len(valid_contours)

    # Draw bounding boxes around valid contours
    for contour in valid_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time people counting result
    cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('People Counting', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Adjust the parameters of the background subtractor and contour filtering based on your environment.

113. **Real-Time Object Recognition and 3D Pose Estimation with OpenCV and MediaPipe:**

```python
# Install the `opencv-python` and `mediapipe` libraries (if not installed): pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Holistic
    results = holistic.process(rgb_frame)

    # Draw landmarks on the face, hands, and pose
    mp_holistic.draw_landmarks(frame, results)

    # Display the real-time object recognition and 3D pose estimation result
    cv2.imshow('Object Recognition and 3D Pose Estimation', frame

)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Ensure that you have the `mediapipe` library installed. You can install it using `pip install mediapipe`.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time object tracking, people counting, and object recognition with OpenCV.





114. **Real-Time Hand Tracking with OpenCV and MediaPipe:**

```python
# Install the `opencv-python` and `mediapipe` libraries (if not installed): pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hand
    results = hands.process(rgb_frame)

    # Draw landmarks and connections on the hands
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_hands.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the real-time hand tracking result
    cv2.imshow('Hand Tracking', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

115. **Real-Time Face Detection and Recognition with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `face_recognition` libraries (if not installed): pip install opencv-python dlib face_recognition

import cv2
import dlib
import face_recognition

# Load known face images and their corresponding names
known_face_images = [face_recognition.load_image_file('path/to/known_face1.jpg'),  # Replace with the path to known face images
                     face_recognition.load_image_file('path/to/known_face2.jpg')]
known_face_names = ['Person 1', 'Person 2']

# Extract face encodings for known faces
known_face_encodings = [face_recognition.face_encodings(img)[0] for img in known_face_images]

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Load the face detector from dlib
face_detector = dlib.get_frontal_face_detector()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame using the dlib face detector
    faces = face_detector(frame)

    # Loop through the detected faces
    for face in faces:
        # Extract face encodings for the current face
        face_encodings = face_recognition.face_encodings(rgb_frame, [face])

        # Check if any of the known faces match the current face
        for i, known_encoding in enumerate(known_face_encodings):
            matches = face_recognition.compare_faces([known_encoding], face_encodings[0])
            name = 'Unknown'

            # If a match is found, assign the name of the known face
            if matches[0]:
                name = known_face_names[i]

            # Draw a rectangle and display the name on the frame
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

    # Display the real-time face detection and recognition result
    cv2.imshow('Face Detection and Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Update the paths to the known face images in the face recognition example.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time hand tracking, face detection, and recognition with OpenCV and external libraries.




114. **Real-Time Hand Tracking with OpenCV and MediaPipe:**

```python
# Install the `opencv-python` and `mediapipe` libraries (if not installed): pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hand
    results = hands.process(rgb_frame)

    # Draw landmarks and connections on the hands
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_hands.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the real-time hand tracking result
    cv2.imshow('Hand Tracking', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

115. **Real-Time Face Detection and Recognition with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `face_recognition` libraries (if not installed): pip install opencv-python dlib face_recognition

import cv2
import dlib
import face_recognition

# Load known face images and their corresponding names
known_face_images = [face_recognition.load_image_file('path/to/known_face1.jpg'),  # Replace with the path to known face images
                     face_recognition.load_image_file('path/to/known_face2.jpg')]
known_face_names = ['Person 1', 'Person 2']

# Extract face encodings for known faces
known_face_encodings = [face_recognition.face_encodings(img)[0] for img in known_face_images]

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Load the face detector from dlib
face_detector = dlib.get_frontal_face_detector()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame using the dlib face detector
    faces = face_detector(frame)

    # Loop through the detected faces
    for face in faces:
        # Extract face encodings for the current face
        face_encodings = face_recognition.face_encodings(rgb_frame, [face])

        # Check if any of the known faces match the current face
        for i, known_encoding in enumerate(known_face_encodings):
            matches = face_recognition.compare_faces([known_encoding], face_encodings[0])
            name = 'Unknown'

            # If a match is found, assign the name of the known face
            if matches[0]:
                name = known_face_names[i]

            # Draw a rectangle and display the name on the frame
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

    # Display the real-time face detection and recognition result
    cv2.imshow('Face Detection and Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Update the paths to the known face images in the face recognition example.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time hand tracking, face detection, and recognition with OpenCV and external libraries.




116. **Real-Time Age and Gender Estimation with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `face_recognition` libraries (if not installed): pip install opencv-python dlib face_recognition

import cv2
import dlib
import face_recognition

# Load the face detector and shape predictor from dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('path/to/shape_predictor_5_face_landmarks.dat')  # Replace with the path to the model file

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces in the frame using the dlib face detector
    faces = face_detector(gray_frame)

    # Loop through the detected faces
    for face in faces:
        # Extract facial landmarks
        landmarks = shape_predictor(gray_frame, face)

        # Extract face landmarks for age and gender estimation
        face_landmarks = face_recognition.face_landmarks(frame, [face])[0]

        # Define the positions of eyes for gender estimation
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']

        # Extract the region of interest for gender estimation
        roi_gender = frame[left_eye[0][1]:right_eye[3][1], left_eye[0][0]:right_eye[3][0]]

        # Perform gender estimation (you can use a pre-trained model or an API)
        # Here, we are using a placeholder function for demonstration purposes
        def estimate_gender(roi):
            return 'Male'  # Placeholder function

        # Define the position of the forehead for age estimation
        forehead = (landmarks.part(0).x, landmarks.part(0).y)

        # Extract the region of interest for age estimation
        roi_age = frame[forehead[1] - 30:forehead[1], forehead[0] - 30:forehead[0] + 30]

        # Perform age estimation (you can use a pre-trained model or an API)
        # Here, we are using a placeholder function for demonstration purposes
        def estimate_age(roi):
            return '25'  # Placeholder function

        # Draw rectangles and display age and gender on the frame
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f'Gender: {estimate_gender(roi_gender)}', (x + 6, y + h + 16), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Age: {estimate_age(roi_age)}', (x + 6, y + h + 32), font, 0.5, (255, 255, 255), 1)

    # Display the real-time age and gender estimation result
    cv2.imshow('Age and Gender Estimation', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the shape predictor model (`shape_predictor_5_face_landmarks.dat`) from the dlib website (http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2) and update the path accordingly. Replace the placeholder functions for gender and age estimation with your preferred methods or models.

117. **Real-Time Object Recognition with OpenCV and YOLO (You Only Look Once):**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Load YOLO model and configuration files
yolo_net = cv2.dnn.readNet('path/to/yolov3.weights', 'path/to/yolov3.cfg')  # Replace with the paths to the YOLO files
yolo_classes = open('path/to/coco.names').read().strip().split('\n')  # Replace with the path to the COCO classes file

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Create a blob from the frame and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    output_layers = yolo_net.getUnconnectedOutLayersNames()
    detections = yolo_net.forward(output_layers)

    # Process the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f'{yolo_classes[class_id]}: {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the real-time object recognition result
    cv2.imshow('Object Recognition (YOLO)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the YOLO files (`yolov3.weights`, `yolov3.cfg`, and `coco.names`) from the official YOLO website (https://pjreddie.com/darknet/yolo/) and update the paths accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time age and gender estimation, and object recognition with OpenCV and deep learning models.




118. **Real-Time Text Detection and Recognition with OpenCV and Tesseract:**

```python
# Install the `opencv-python` and `pytesseract` libraries (if not installed): pip install opencv-python pytesseract

import cv2
import pytesseract

# Set the path to the Tesseract executable (replace with your own path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance text regions
    _, thresholded = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

    # Use Tesseract to perform OCR on the thresholded image
    text = pytesseract.image_to_string(thresholded)

    # Display the real-time text detection and recognition result
    cv2.putText(frame, f'Text: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Text Detection and Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Make sure to install Tesseract OCR on your system and update the Tesseract executable path accordingly.

119. **Real-Time Lane Detection and Tracking with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Define a region of interest (ROI) for lane detection
    roi_vertices = np.array([[(50, frame.shape[0]), (frame.shape[1] // 2 - 50, frame.shape[0] // 2 + 50),
                              (frame.shape[1] // 2 + 50, frame.shape[0] // 2 + 50), (frame.shape[1] - 50, frame.shape[0])]],
                            dtype=np.int32)
    roi_frame = np.zeros_like(edges)
    cv2.fillPoly(roi_frame, roi_vertices, 255)
    roi_edges = cv2.bitwise_and(edges, roi_frame)

    # Perform Hough transform to detect lines in the ROI
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw detected lines on the frame
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the real-time lane detection and tracking result
    cv2.imshow('Lane Detection and Tracking', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

120. **Real-Time Human Pose Estimation with OpenCV and PoseNet:**

```python
# Install the `opencv-python` and `posenet` libraries (if not installed): pip install opencv-python posenet

import cv2
from posenet import PoseNet

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Initialize PoseNet
posenet = PoseNet()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Perform human pose estimation using PoseNet
    keypoints, _ = posenet.process_image(frame)

    # Draw keypoints and skeletons on the frame
    posenet.draw_keypoints(frame, keypoints)
    posenet.draw_skeleton(frame, keypoints)

    # Display the real-time human pose estimation result
    cv2.imshow('Human Pose Estimation (PoseNet)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Install the `posenet` library using `pip install posenet`.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time text detection and recognition, lane detection and tracking, and human pose




118. **Real-Time Text Detection and Recognition with OpenCV and Tesseract:**

```python
# Install the `opencv-python` and `pytesseract` libraries (if not installed): pip install opencv-python pytesseract

import cv2
import pytesseract

# Set the path to the Tesseract executable (replace with your own path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance text regions
    _, thresholded = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

    # Use Tesseract to perform OCR on the thresholded image
    text = pytesseract.image_to_string(thresholded)

    # Display the real-time text detection and recognition result
    cv2.putText(frame, f'Text: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Text Detection and Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Make sure to install Tesseract OCR on your system and update the Tesseract executable path accordingly.

119. **Real-Time Lane Detection and Tracking with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Define a region of interest (ROI) for lane detection
    roi_vertices = np.array([[(50, frame.shape[0]), (frame.shape[1] // 2 - 50, frame.shape[0] // 2 + 50),
                              (frame.shape[1] // 2 + 50, frame.shape[0] // 2 + 50), (frame.shape[1] - 50, frame.shape[0])]],
                            dtype=np.int32)
    roi_frame = np.zeros_like(edges)
    cv2.fillPoly(roi_frame, roi_vertices, 255)
    roi_edges = cv2.bitwise_and(edges, roi_frame)

    # Perform Hough transform to detect lines in the ROI
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw detected lines on the frame
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the real-time lane detection and tracking result
    cv2.imshow('Lane Detection and Tracking', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

120. **Real-Time Human Pose Estimation with OpenCV and PoseNet:**

```python
# Install the `opencv-python` and `posenet` libraries (if not installed): pip install opencv-python posenet

import cv2
from posenet import PoseNet

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Initialize PoseNet
posenet = PoseNet()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Perform human pose estimation using PoseNet
    keypoints, _ = posenet.process_image(frame)

    # Draw keypoints and skeletons on the frame
    posenet.draw_keypoints(frame, keypoints)
    posenet.draw_skeleton(frame, keypoints)

    # Display the real-time human pose estimation result
    cv2.imshow('Human Pose Estimation (PoseNet)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Install the `posenet` library using `pip install posenet`.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time text detection and recognition, lane detection and tracking, and human pose

estimation with OpenCV and various libraries.




121. **Real-Time Facial Landmark Detection with OpenCV and Dlib:**

```python
# Install the `opencv-python` and `dlib` libraries (if not installed): pip install opencv-python dlib

import cv2
import dlib

# Load the face detector and facial landmark predictor from dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('path/to/shape_predictor_68_face_landmarks.dat')  # Replace with the path to the model file

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces in the frame using the dlib face detector
    faces = face_detector(gray_frame)

    # Loop through the detected faces
    for face in faces:
        # Extract facial landmarks
        landmarks = landmark_predictor(gray_frame, face)

        # Draw circles around each facial landmark
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display the real-time facial landmark detection result
    cv2.imshow('Facial Landmark Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the facial landmark predictor model file (`shape_predictor_68_face_landmarks.dat`) from the dlib website (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and update the path accordingly.

122. **Real-Time Object Tracking with OpenCV and MOSSE Tracker:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

# Create the MOSSE tracker
tracker = cv2.TrackerMOSSE_create()

# Read the first frame
ret, frame = cap.read()

# Select a bounding box around the object to be tracked
bbox = cv2.selectROI(frame, False)

# Initialize the tracker with the bounding box
tracker.init(frame, bbox)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Update the tracker
    success, bbox = tracker.update(frame)

    # Draw the bounding box on the frame
    if success:
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time object tracking result
    cv2.imshow('Object Tracking (MOSSE)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

123. **Real-Time Barcode and QR Code Detection with OpenCV and ZBar:**

```python
# Install the `opencv-python` and `zbarlight` libraries (if not installed): pip install opencv-python zbarlight

import cv2
from PIL import Image
from zbarlight import scan_codes

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use ZBar to detect barcodes and QR codes
    codes = scan_codes(['qrcode'], Image.fromarray(gray_frame))

    # Draw rectangles around detected codes
    if codes:
        for code in codes:
            (x, y, w, h) = code['rect']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time barcode and QR code detection result
    cv2.imshow('Barcode and QR Code Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Install the `zbarlight` library using `pip install zbarlight`.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time facial landmark detection, object tracking, and barcode/QR code detection with OpenCV and various libraries.




124. **Real-Time Document Scanner with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Define the document corners
document_corners = np.array([[50, 50], [50, 400], [400, 400], [400, 50]], dtype=np.float32)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Find contours in the edge-detected frame
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    # Sort contours based on area in descending order
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    # Find the largest contour
    if valid_contours:
        largest_contour = valid_contours[0]

        # Approximate the polygonal curve of the largest contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Ensure that the polygon has four corners
        if len(approx_polygon) == 4:
            # Draw the contours on the frame
            cv2.drawContours(frame, [approx_polygon], -1, (0, 255, 0), 2)

            # Convert the approximated polygon to float32
            approx_polygon = np.array(approx_polygon, dtype=np.float32)

            # Perform perspective transformation to obtain a top-down view
            transformation_matrix = cv2.getPerspectiveTransform(approx_polygon, document_corners)
            scanned_document = cv2.warpPerspective(frame, transformation_matrix, (frame.shape[1], frame.shape[0]))

            # Display the real-time document scanning result
            cv2.imshow('Document Scanner', scanned_document)
        else:
            # If the polygon doesn't have four corners, reset the scanner
            cv2.drawContours(frame, [approx_polygon], -1, (0, 0, 255), 2)

    # Display the original frame with contours
    cv2.imshow('Document Scanner (Original)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

125. **Real-Time Object Detection and Tracking with OpenCV and SORT:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
from sort import Sort

# Clone or download the SORT algorithm from: https://github.com/abewley/sort
# Extract the contents and navigate to the 'sort' directory
# Install the 'sort' module: python setup.py install

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Create the SORT tracker
tracker = Sort()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Perform object detection (you can replace this with your preferred object detection method)
    # Here, we are using a simple example of detecting a blue object
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract object detections as bounding boxes [x, y, w, h]
    detections = [cv2.boundingRect(contour) for contour in contours]

    # Update the SORT tracker with current detections
    trackers = tracker.update(np.array(detections))

    # Draw bounding boxes for tracked objects
    for bbox in trackers:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time object detection and tracking result
    cv2.imshow('Object Detection and Tracking (SORT)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the SORT algorithm from the official repository (https://github.com/abewley/sort) and follow the installation instructions. Replace the object detection part with your preferred object detection method.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time document scanning, and object detection and tracking with OpenCV and additional libraries.




126. **Real-Time Image Segmentation with OpenCV and GrabCut:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Initialize GrabCut parameters
rect = (50, 50, 400, 400)  # Initial rectangle coordinates (x, y, width, height)
mask = np.zeros((cap.get(4), cap.get(3)), dtype=np.uint8)
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Initialize the mask with the rectangle and perform GrabCut
    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to obtain a binary mask for foreground and background
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the binary mask to the original frame
    segmented_frame = frame * mask2[:, :, np.newaxis]

    # Display the real-time image segmentation result
    cv2.imshow('Image Segmentation (GrabCut)', segmented_frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

127. **Real-Time Color Detection with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for detection (here, detecting green color)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a binary mask for the specified color range
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Apply the mask to the original frame
    result_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the real-time color detection result
    cv2.imshow('Color Detection', result_frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

128. **Real-Time Hand Gesture Recognition with OpenCV and Mediapipe:**

```python
# Install the `opencv-python` and `mediapipe` libraries (if not installed): pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hand
    results = hands.process(rgb_frame)

    # Check for hand landmarks and recognize gestures
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            hand_landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in landmarks.landmark]

            # Perform gesture recognition (customize as needed)
            # Here, we check if the index finger is pointing up
            index_finger_tip = hand_landmarks[8]
            index_finger_base = hand_landmarks[5]
            if index_finger_tip[1] < index_finger_base[1]:
                cv2.putText(frame, 'Index Finger Up', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw landmarks on the frame
            mp_hands.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the real-time hand gesture recognition result
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Ensure that you have the `mediapipe` library installed. You can install it using `pip install mediapipe`.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time image segmentation, color detection, and hand gesture recognition with OpenCV and various libraries.




129. **Real-Time Emotion Detection with OpenCV and Deep Learning:**

```python
# Install the `opencv-python` and `tensorflow` libraries (if not installed): pip install opencv-python tensorflow

import cv2
import tensorflow as tf

# Load the pre-trained emotion detection model
emotion_model = tf.keras.models.load_model('path/to/emotion_model.h5')  # Replace with the path to your model file

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Set the font and scale for displaying emotions
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
color = (255, 255, 255)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame for emotion detection model input
    resized_frame = cv2.resize(gray_frame, (48, 48))
    resized_frame = resized_frame / 255.0  # Normalize pixel values

    # Reshape the frame for model prediction
    input_data = resized_frame.reshape((1, 48, 48, 1))

    # Perform emotion prediction
    predictions = emotion_model.predict(input_data)

    # Get the predicted emotion
    predicted_emotion = emotions[np.argmax(predictions)]

    # Display the predicted emotion on the frame
    cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), font, scale, color, 2)

    # Display the real-time emotion detection result
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Replace `'path/to/emotion_model.h5'` with the path to your pre-trained emotion detection model file.

130. **Real-Time Face Recognition with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `face_recognition` libraries (if not installed): pip install opencv-python dlib face_recognition

import cv2
import dlib
import face_recognition

# Load the known faces and their corresponding names
known_faces = []
known_names = []

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Find faces in the frame using the dlib face detector
    faces = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, faces)

    # Loop through the detected faces
    for (top, right, bottom, left), face_encoding in zip(faces, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        name = 'Unknown'

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Draw rectangles and display the name on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the real-time face recognition result
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Add known faces and their corresponding names to the `known_faces` and `known_names` lists. You can use the `face_recognition.load_image_file` function to load known faces.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time emotion detection, and face recognition with OpenCV and deep learning models.




131. **Real-Time Object Detection with OpenCV and YOLO (You Only Look Once):**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2

# Load the YOLO model and configuration files
net = cv2.dnn.readNet('path/to/yolov3.weights', 'path/to/yolov3.cfg')  # Replace with the path to your YOLO files
classes = []
with open('path/to/coco.names', 'r') as f:  # Replace with the path to your COCO names file
    classes = [line.strip() for line in f.readlines()]

# Capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify the path to a video file

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Prepare the frame for YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Perform YOLO object detection
    detections = net.forward(output_layer_names)

    # Process and display the detected objects
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Set the confidence threshold
                # Calculate the coordinates of the bounding box
                x, y, w, h = int(obj[0] * width), int(obj[1] * height), int(obj[2] * width), int(obj[3] * height)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f'{classes[class_id]}: {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the real-time object detection result
    cv2.imshow('Object Detection (YOLO)', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Replace `'path/to/yolov3.weights'`, `'path/to/yolov3.cfg'`, and `'path/to/coco.names'` with the paths to your YOLO files.

132. **Real-Time Age and Gender Estimation with OpenCV and Pre-trained Models:**

```python
# Install the `opencv-python` and `opencv-contrib-python` libraries (if not installed): pip install opencv-python opencv-contrib-python

import cv2
import numpy as np

# Load the pre-trained age and gender estimation models
age_net = cv2.dnn.readNet('path/to/age_net.caffemodel', 'path/to/deploy_age.prototxt')  # Replace with the paths to your models
gender_net = cv2.dnn.readNet('path/to/gender_net.caffemodel', 'path/to/deploy_gender.prototxt')

# Define the age and gender classes
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classes = ['Male', 'Female']

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Prepare the frame for age and gender estimation
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    gender_net.setInput(blob)

    # Perform age and gender estimation
    age_preds = age_net.forward()
    gender_preds = gender_net.forward()

    # Get the predicted age and gender
    age = age_classes[np.argmax(age_preds)]
    gender = gender_classes[np.argmax(gender_preds)]

    # Display the age and gender on the frame
    label = f'Age: {age}, Gender: {gender}'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the real-time age and gender estimation result
    cv2.imshow('Age and Gender Estimation', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Replace `'path/to/age_net.caffemodel'`, `'path/to/deploy_age.prototxt'`, `'path/to/gender_net.caffemodel'`, and `'path/to/deploy_gender.prototxt'` with the paths to your age and gender estimation models.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time object detection with YOLO and age and gender estimation with OpenCV and pre-trained models.





133. **Real-Time Lane Detection with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Define a region of interest (ROI) for lane detection
    roi_vertices = np.array([[0, frame.shape[0]], [frame.shape[1] // 2, frame.shape[0] // 2], [frame.shape[1], frame.shape[0]]], dtype=np.int32)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, [roi_vertices], 255)
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    # Perform Hough line transformation to detect lines in the ROI
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the real-time lane detection result
    cv2.imshow('Lane Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

134. **Real-Time Optical Flow with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()

# Convert the first frame to grayscale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    # Read the current frame
    ret, frame2 = cap.read()

    # Convert the current frame to grayscale
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow using the Farneback method
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude and angle of the optical flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set the hue according to the optical flow direction
    hsv[..., 0] = ang * 180 / np.pi / 2

    # Set the value according to the optical flow magnitude
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for visualization
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Display the real-time optical flow result
    cv2.imshow('Optical Flow', flow_rgb)

    # Update the previous frame
    prvs = next_frame

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

135. **Real-Time Face Swapping with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `numpy` libraries (if not installed): pip install opencv-python dlib numpy

import cv2
import dlib
import numpy as np

# Load the face and landmark detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('path/to/shape_predictor_68_face_landmarks.dat')  # Replace with the path to the model file

# Load the source and target images
source_image = cv2.imread('path/to/source_image.jpg')  # Replace with the path to the source image
target_image = cv2.imread('path/to/target_image.jpg')  # Replace with the path to the target image

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray_frame)

    # Loop through the detected faces
    for face in faces:
        # Extract facial landmarks
        landmarks = predictor(gray_frame, face)

        # Convert landmarks to NumPy array
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract the region of interest (ROI) around the mouth area
        mouth_roi = frame[landmarks_np[48:68, 1].min():landmarks_np[48:68, 1].max(), landmarks_np[48:68, 0].min():landmarks_np[48:68, 0].max()]

        # Resize the source image to match the size of the mouth ROI
        source_resized = cv2.resize(source_image, (mouth_roi.shape[1], mouth_roi.shape[0]))

        # Swap faces by replacing the mouth ROI with the source image
        frame[landmarks_np[48:68, 1].min():landmarks_np[48:68, 1].max(), landmarks_np[48:68, 0].min():landmarks_np[48:68, 0].max()] = source_resized

    # Display the real-time face swapping result
    cv2.imshow('Face Swapping', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the facial landmark predictor model file (`shape_predictor_68_face_landmarks.dat`) from the dlib website (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and update the path accordingly. Replace `'path/to/source_image.jpg'` and `'path/to/target_image.jpg'

` with the paths to your source and target images.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time lane detection, optical flow, and face swapping with OpenCV and various libraries.





136. **Real-Time Style Transfer with OpenCV and Neural Style Transfer:**

```python
# Install the `opencv-python`, `numpy`, and `torch` libraries (if not installed): pip install opencv-python numpy torch

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import vgg19
from PIL import Image

# Load the pre-trained VGG19 model
vgg = vgg19(pretrained=True).features.eval()

# Define the style and content image paths
style_path = 'path/to/style_image.jpg'  # Replace with the path to the style image
content_path = 'path/to/content_image.jpg'  # Replace with the path to the content image

# Load the style and content images
style_image = Image.open(style_path)
content_image = Image.open(content_path)

# Define the image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Preprocess the style and content images
style_tensor = transform(style_image)
content_tensor = transform(content_image)

# Create a function to convert a tensor to an image
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().numpy().squeeze()
    tensor = tensor.transpose(1, 2, 0)
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

# Define the Gram matrix calculation function
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Define the style and content layers for style transfer
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
content_layer = 'conv4_2'

# Initialize the generated image as the content image
generated_image = content_tensor.clone().detach().requires_grad_(True)

# Define the optimizer and loss function
optimizer = torch.optim.Adam([generated_image], lr=0.01)
mse_loss = torch.nn.MSELoss()

# Perform neural style transfer
num_iterations = 500
for i in range(num_iterations):
    optimizer.zero_grad()

    # Forward pass to extract features
    features_style = {}
    features_content = {}

    x_style = style_tensor.unsqueeze(0)
    x_content = generated_image.unsqueeze(0)

    for name, layer in vgg._modules.items():
        x_style = layer(x_style)
        x_content = layer(x_content)

        if name in style_layers:
            features_style[name] = gram_matrix(x_style)

        if name == content_layer:
            features_content[name] = x_content

    # Calculate the style loss
    style_loss = 0
    for layer in style_layers:
        style_loss += mse_loss(features_style[layer], gram_matrix(x_content))

    # Calculate the content loss
    content_loss = mse_loss(features_content[content_layer], x_content)

    # Calculate the total loss
    alpha = 1  # Style weight
    beta = 1e6  # Content weight
    total_loss = alpha * style_loss + beta * content_loss

    # Backward pass and optimization step
    total_loss.backward()
    optimizer.step()

    # Print the progress
    if i % 50 == 0:
        print(f"Iteration {i}/{num_iterations}, Total Loss: {total_loss.item()}")

# Convert the generated tensor to an image
generated_image = tensor_to_image(generated_image.detach())
generated_image.show()
```

Note: Replace `'path/to/style_image.jpg'` and `'path/to/content_image.jpg'` with the paths to your style and content images.

137. **Real-Time Image Captioning with OpenCV and a Pre-trained Model:**

```python
# Install the `opencv-python`, `torch`, `torchvision`, `numpy`, and `nltk` libraries (if not installed): pip install opencv-python torch torchvision numpy nltk

import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define COCO class labels
coco_labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'


]

# Load the COCO model's transform
transform = F.Compose([
    F.ToTensor()
])

# Function to perform image captioning
def image_captioning(image_path):
    # Load and transform the input image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)

    # Perform inference using the Mask R-CNN model
    with torch.no_grad():
        prediction = model([img_tensor])

    # Get the predicted labels and boxes
    labels = prediction[0]['labels']
    boxes = prediction[0]['boxes']

    # Display the image with bounding boxes and labels
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    font = ImageFont.load_default()

    for label, box in zip(labels, boxes):
        label_str = coco_labels[label]
        draw.rectangle(box.tolist(), outline="red", width=2)
        draw.text((box[0], box[1]), f"{label_str}", fill="red", font=font)

    img_draw.show()

    # Generate a caption based on detected objects
    detected_objects = [coco_labels[label] for label in labels]
    caption = f"This image contains {', '.join(detected_objects[:-1])}, and {detected_objects[-1]}."

    return caption

# Example usage
image_path = 'path/to/image.jpg'  # Replace with the path to your image
caption_result = image_captioning(image_path)
print("Generated Caption:", caption_result)
```

Note: Replace `'path/to/image.jpg'` with the path to your input image.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time style transfer, image captioning with OpenCV, and various pre-trained models.




138. **Real-Time Hand Gesture Recognition with OpenCV and MediaPipe:**

```python
# Install the `opencv-python`, `mediapipe`, and `numpy` libraries (if not installed): pip install opencv-python mediapipe numpy

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Get the hand gesture (example: check if the thumb is up)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Define a threshold for thumb-up gesture
            threshold = 0.8

            # Check if the thumb is up based on the y-coordinate of thumb tip
            is_thumb_up = thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y and thumb_tip.y < threshold * h

            # Display hand gesture status
            gesture_status = "Thumb Up" if is_thumb_up else "No Gesture"
            cv2.putText(frame, f'Gesture: {gesture_status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the real-time hand gesture recognition result
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

139. **Real-Time Document Scanner with OpenCV:**

```python
# Install the `opencv-python` library (if not installed): pip install opencv-python

import cv2
import numpy as np

# Capture video from a webcam
cap = cv2.VideoCapture(0)

# Create a blank image to store the scanned document
scanned_document = np.zeros((500, 500, 3), dtype=np.uint8)

# Initialize the document scanner
scanner = cv2.QRCodeDetector()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours based on their area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Loop over the contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has four vertices, it is likely a document
        if len(approx) == 4:
            # Draw the contour on the frame
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

            # Apply a perspective transform to obtain a top-down view of the document
            pts = np.float32(approx)
            scanned_document = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])), (500, 500))

            # Break the loop if a document is found
            break

    # Display the real-time document scanner result
    cv2.imshow('Document Scanner', np.hstack([frame, scanned_document]))

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

140. **Real-Time Facial Landmark Detection with OpenCV and Dlib:**

```python
# Install the `opencv-python`, `dlib`, and `numpy` libraries (if not installed): pip install opencv-python dlib numpy

import cv2
import dlib
import numpy as np

# Load the face and landmark detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('path/to/shape_predictor_68_face_landmarks.dat')  # Replace with the path to the model file

# Capture video from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray_frame)

    # Loop through the detected faces
    for face in faces:
        # Extract facial landmarks
        landmarks = predictor(gray_frame, face)

        # Convert landmarks to NumPy array
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Draw facial landmarks on the frame
        for (x, y) in landmarks_np:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Draw a rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time facial landmark

detection result
    cv2.imshow('Facial Landmark Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Note: Download the facial landmark predictor model file (`shape_predictor_68_face_landmarks.dat`) from the dlib website (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and update the path accordingly.

Feel free to experiment with and customize these examples based on your specific needs. These cover real-time hand gesture recognition, document scanning, and facial landmark detection with OpenCV and various libraries.
