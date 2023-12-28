import cv2
import matplotlib.pyplot as plt

def display_images(images, window_titles=None, window_size=(800, 600)):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    for image, title in zip(images, window_titles):
        key = display_image(image, window_title=title, window_size=window_size)
        user_inputs.append((title, key))

    return user_inputs

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

image_paths = ['images/image.jpg', 'images/image.jpeg']
images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    user_inputs_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'])
    user_inputs_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'])

    prompt_user()

    for title, key in user_inputs_original:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, key in user_inputs_grayscale:
        print(f"Key pressed for {title}: {chr(key)}")
else:
    print("Error: Unable to load one or more images.")