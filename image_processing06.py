import cv2
import matplotlib.pyplot as plt

def display_images(image_path):
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_grayscale, cmap='gray')
    plt.title('Grayscale Image')

    cv2.imwrite('images/src/grayscale_image06.jpg', grayscale_image)

    plt.show(block=False)

    while True:
        key = cv2.waitKey(1) 

        if key == 27:  
            print("Window closed by user.")
            break
        elif key != -1:
            print(f"Key pressed: {chr(key)}")

    plt.close()

image_path = 'images/image.jpg'
display_images(image_path)