import cv2

image_path = 'images/image.jpg'  
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error: Unable to load the image at {image_path}")
else:
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original Image', original_image)
    cv2.imshow('Grayscale Image', grayscale_image)

    cv2.imwrite('images/grayscale_image.jpg', grayscale_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()