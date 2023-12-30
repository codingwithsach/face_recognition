import cv2
import matplotlib.pyplot as plt

def display_image(image, window_title='Image', window_size=(800, 600)):
    resized_image = cv2.resize(image, window_size)

    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    plt.show(block=False)

    key = cv2.waitKey(0) & 0xFF

    plt.close()

    return key

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

image_path = 'images/image.jpg'
original_image = cv2.imread(image_path)

if original_image is not None:
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    key_original = display_image(original_image, window_title='Original Image', window_size=(800, 600))
    key_grayscale = display_image(grayscale_image, window_title='Grayscale Image', window_size=(800, 600))

    prompt_user()

    print(f"Key pressed for original image: {chr(key_original)}")
    print(f"Key pressed for grayscale image: {chr(key_grayscale)}")
else:
    print(f"Error: Unable to load the image at {image_path}")