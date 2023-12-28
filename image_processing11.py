import cv2
import matplotlib.pyplot as plt

def display_image(image, window_title='Image', window_size=(800, 600), wait_key=True):
    resized_image = cv2.resize(image, window_size)

    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
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

    display_image(original_image, window_title='Original Image', window_size=(800, 600))
    display_image(grayscale_image, window_title='Grayscale Image', window_size=(800, 600))
else:
    print(f"Error: Unable to load the image at {image_path}")