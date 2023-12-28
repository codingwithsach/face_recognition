import cv2
import matplotlib.pyplot as plt

image_path = 'images/image.jpg'
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error: Unable to load the image at {image_path}")
else:
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')

    cv2.imwrite('images/grayscale_image0.jpg', grayscale_image)

    plt.show()