import cv2
import matplotlib.pyplot as plt

def display_images(images, window_titles=None, window_size=(800, 600)):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0 
    comments = {}

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size)

        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            break
        elif chr(key).lower() == 'c':
            comment = input(f"Enter a comment for {title}: ")
            comments[title] = comment

        user_inputs.append((title, key))

    return user_inputs, comments

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

    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'])
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'])

    prompt_user()
    for title, key in user_inputs_original:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, key in user_inputs_grayscale:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments_original.items():
        print(f"Comment for {title}: {comment}")

    for title, comment in comments_grayscale.items():
        print(f"Comment for {title}: {comment}")
else:
    print("Error: Unable to load one or more images.")