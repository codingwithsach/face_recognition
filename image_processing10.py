import cv2
import matplotlib.pyplot as plt

def display_image(image, window_title='Image', wait_key=True):
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    plt.show(block=False)

    while cv2.getWindowProperty(window_title, 0) >= 0:
        if wait_key:
            key = cv2.waitKey(1) & 0xFF 

            if key == 27: 
                print("Window closed by user.")
                break
            elif key != 255: 
                print(f"Key pressed: {chr(key)}")
        else:
            cv2.waitKey(0)
            break

    plt.close()

image_path = 'images/image.jpg'
original_image = cv2.imread(image_path)

if original_image is not None:
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    display_image(resized_original, window_title='Original Image')
    display_image(resized_grayscale, window_title='Grayscale Image')
else:
    print(f"Error: Unable to load the image at {image_path}")