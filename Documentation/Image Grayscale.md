Certainly! Converting an image to grayscale using OpenCV is a common task in computer vision. Below is a step-by-step Python code using OpenCV to achieve this:

```python
import cv2

# Step 1: Read the image
image_path = 'path/to/your/image.jpg'  # Replace with the actual path to your image file
original_image = cv2.imread(image_path)

# Step 2: Check if the image is loaded successfully
if original_image is None:
    print(f"Error: Unable to load the image at {image_path}")
else:
    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Display the original and grayscale images
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Grayscale Image', grayscale_image)

    # Step 5: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 6: Wait for a key event and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

Make sure to replace `'path/to/your/image.jpg'` with the actual path to your image file. This code reads the image, converts it to grayscale, displays both the original and grayscale images, and optionally saves the grayscale image to a file.

Note: If you're working in a Jupyter Notebook or another environment where showing images directly is supported, you may use `plt.imshow()` from the `matplotlib` library for better visualization.

Certainly! Continuing from where we left off, let's further enhance the code to use `matplotlib` for displaying images if you're working in an environment that supports it:

```python
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image
image_path = 'path/to/your/image.jpg'  # Replace with the actual path to your image file
original_image = cv2.imread(image_path)

# Step 2: Check if the image is loaded successfully
if original_image is None:
    print(f"Error: Unable to load the image at {image_path}")
else:
    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')

    # Step 5: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 6: Show the images
    plt.show()
```

In this code, `plt.imshow()` is used to display the images, and `plt.subplot()` is used to arrange the original and grayscale images side by side. The `cmap='gray'` argument is specified for the grayscale image to ensure it is displayed correctly.

Remember to install the necessary libraries if you haven't already:

```bash
pip install opencv-python matplotlib
```

Replace `'path/to/your/image.jpg'` with the actual path to your image file. Now, when you run the code, it should display both the original and grayscale images in a single figure using matplotlib.

Certainly! Continuing from the previous code, I'll add a few more comments to explain the code and show how to close the plot window after displaying the images:

```python
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image
image_path = 'path/to/your/image.jpg'  # Replace with the actual path to your image file
original_image = cv2.imread(image_path)

# Step 2: Check if the image is loaded successfully
if original_image is None:
    print(f"Error: Unable to load the image at {image_path}")
else:
    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')

    # Step 5: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 6: Show the images
    plt.show()

    # Step 7: Wait for a key event and then close the plot window (optional)
    plt.waitforbuttonpress()
    plt.close()
```

In Step 7, `plt.waitforbuttonpress()` pauses the execution until a key is pressed in the plot window. After a key is pressed, `plt.close()` closes the plot window. This step is optional and is useful if you want to keep the plot window open until you decide to close it.

Remember to replace `'path/to/your/image.jpg'` with the actual path to your image file. If you're running this code in a script or an interactive environment, it should display the images, wait for a key press, and then close the plot window.

Certainly! Continuing from the previous code, let's add a few more details, including handling the case where the user closes the plot window:

```python
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image
image_path = 'path/to/your/image.jpg'  # Replace with the actual path to your image file
original_image = cv2.imread(image_path)

# Step 2: Check if the image is loaded successfully
if original_image is None:
    print(f"Error: Unable to load the image at {image_path}")
else:
    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')

    # Step 5: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 6: Show the images
    plt.show()

    # Step 7: Wait for a key event and handle window closure (optional)
    key = cv2.waitKey(0)
  
    if key == 27:  # 27 is the ASCII code for the Esc key
        print("Window closed by user.")
    else:
        print(f"Key pressed: {chr(key)}")

    # Step 8: Close the plot window
    plt.close()
```

In Step 7, `cv2.waitKey(0)` waits for a key event. If the user presses the Esc key (ASCII code 27), it prints a message indicating that the window was closed by the user. Otherwise, it prints the ASCII code of the key pressed.

Make sure to replace `'path/to/your/image.jpg'` with the actual path to your image file. This code will display the images, wait for a key press, and provide feedback on whether the window was closed by the user or if a key was pressed.

Certainly! Continuing from the previous code, let's add a bit more functionality, such as resizing the displayed images for better visualization:

```python
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image
image_path = 'path/to/your/image.jpg'  # Replace with the actual path to your image file
original_image = cv2.imread(image_path)

# Step 2: Check if the image is loaded successfully
if original_image is None:
    print(f"Error: Unable to load the image at {image_path}")
else:
    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Resize images for better visualization
    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    # Step 5: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_grayscale, cmap='gray')
    plt.title('Grayscale Image')

    # Step 6: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 7: Show the images
    plt.show()

    # Step 8: Wait for a key event and handle window closure (optional)
    key = cv2.waitKey(0)
  
    if key == 27:  # 27 is the ASCII code for the Esc key
        print("Window closed by user.")
    else:
        print(f"Key pressed: {chr(key)}")

    # Step 9: Close the plot window
    plt.close()
```

In this version, Step 4 resizes the original and grayscale images using `cv2.resize()` for better visualization. You can adjust the dimensions `(800, 600)` to suit your preference.

Remember to replace `'path/to/your/image.jpg'` with the actual path to your image file. This code will display resized versions of the original and grayscale images, wait for a key press, and provide feedback on whether the window was closed by the user or if a key was pressed.

Certainly! Let's continue by adding a few more improvements, such as handling the case where the user closes the window with the close button, and encapsulating the code in a function for better organization:

```python
import cv2
import matplotlib.pyplot as plt

def display_images(image_path):
    # Step 1: Read the image
    original_image = cv2.imread(image_path)

    # Step 2: Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return

    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Resize images for better visualization
    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    # Step 5: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_grayscale, cmap='gray')
    plt.title('Grayscale Image')

    # Step 6: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 7: Show the images
    plt.show()

    # Step 8: Wait for a key event and handle window closure (optional)
    key = cv2.waitKey(0)
  
    if key == 27:  # 27 is the ASCII code for the Esc key
        print("Window closed by user.")
    else:
        print(f"Key pressed: {chr(key)}")

    # Step 9: Close the plot window
    plt.close()

# Example usage:
image_path = 'path/to/your/image.jpg'
display_images(image_path)
```

In this version, the code is encapsulated in a function named `display_images`, making it more modular and easier to reuse. You can call this function with different image paths to display and process multiple images.




Certainly! Let's continue by addressing the issue of the program not terminating properly if the user closes the window using the close button. We'll use the `plt.show(block=False)` method to ensure that the window is non-blocking, allowing the program to continue running even when the window is open. Additionally, we'll add a loop to check for user input or window closure:

```python
import cv2
import matplotlib.pyplot as plt

def display_images(image_path):
    # Step 1: Read the image
    original_image = cv2.imread(image_path)

    # Step 2: Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return

    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Resize images for better visualization
    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    # Step 5: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_grayscale, cmap='gray')
    plt.title('Grayscale Image')

    # Step 6: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 7: Show the images (non-blocking)
    plt.show(block=False)

    # Step 8: Loop to check for user input or window closure
    while True:
        key = cv2.waitKey(1)  # Check for key events every 1 ms

        if key == 27:  # 27 is the ASCII code for the Esc key
            print("Window closed by user.")
            break
        elif key != -1:
            print(f"Key pressed: {chr(key)}")

    # Step 9: Close the plot window
    plt.close()

# Example usage:
image_path = 'path/to/your/image.jpg'
display_images(image_path)
```

In this version, the `plt.show(block=False)` is used to make the plot window non-blocking, and a `while` loop is added to continuously check for key events or window closure. The loop exits when the user presses the Esc key or closes the window. This way, the program will terminate gracefully even if the user closes the window using the close button.




Certainly! Let's continue by addressing the issue of the program not terminating properly if the user closes the window using the close button. We'll use the `plt.show(block=False)` method to ensure that the window is non-blocking, allowing the program to continue running even when the window is open. Additionally, we'll add a loop to check for user input or window closure:

```python
import cv2
import matplotlib.pyplot as plt

def display_images(image_path):
    # Step 1: Read the image
    original_image = cv2.imread(image_path)

    # Step 2: Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return

    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Resize images for better visualization
    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    # Step 5: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_grayscale, cmap='gray')
    plt.title('Grayscale Image')

    # Step 6: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 7: Show the images (non-blocking)
    plt.show(block=False)

    # Step 8: Loop to check for user input or window closure
    while True:
        key = cv2.waitKey(1)  # Check for key events every 1 ms

        if key == 27:  # 27 is the ASCII code for the Esc key
            print("Window closed by user.")
            break
        elif key != -1:
            print(f"Key pressed: {chr(key)}")

    # Step 9: Close the plot window
    plt.close()

# Example usage:
image_path = 'path/to/your/image.jpg'
display_images(image_path)
```

In this version, the `plt.show(block=False)` is used to make the plot window non-blocking, and a `while` loop is added to continuously check for key events or window closure. The loop exits when the user presses the Esc key or closes the window. This way, the program will terminate gracefully even if the user closes the window using the close button.



Certainly! Continuing from the previous code, let's add an option to close the window by clicking on the close button. Additionally, we'll add comments to explain the new changes:

```python
import cv2
import matplotlib.pyplot as plt

def display_images(image_path):
    # Step 1: Read the image
    original_image = cv2.imread(image_path)

    # Step 2: Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return

    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Resize images for better visualization
    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    # Step 5: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_grayscale, cmap='gray')
    plt.title('Grayscale Image')

    # Step 6: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 7: Show the images (non-blocking)
    plt.show(block=False)

    # Step 8: Loop to check for user input or window closure
    while True:
        key = cv2.waitKey(1)  # Check for key events every 1 ms

        if key == 27:  # 27 is the ASCII code for the Esc key
            print("Window closed by user.")
            break
        elif key != -1:
            print(f"Key pressed: {chr(key)}")

        # Check if the plot window was closed using the close button
        if not plt.fignum_exists(1):
            print("Window closed by clicking on the close button.")
            break

    # Step 9: Close the plot window
    plt.close()

# Example usage:
image_path = 'path/to/your/image.jpg'
display_images(image_path)
```

In this version, we use `plt.fignum_exists(1)` to check if the plot window with the figure number 1 (our display) still exists. If it doesn't exist, it means the user closed the window using the close button, and the loop will break. This ensures that the program gracefully handles both user input and window closure.





Certainly! Let's continue by improving the code to handle potential issues with keyboard events and provide a more flexible way to close the window. We'll use the `cv2.getWindowProperty` function to check if the window is still open:

```python
import cv2
import matplotlib.pyplot as plt

def display_images(image_path):
    # Step 1: Read the image
    original_image = cv2.imread(image_path)

    # Step 2: Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return

    # Step 3: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Resize images for better visualization
    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    # Step 5: Display the original and grayscale images using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_grayscale, cmap='gray')
    plt.title('Grayscale Image')

    # Step 6: Save the grayscale image (optional)
    cv2.imwrite('path/to/save/grayscale_image.jpg', grayscale_image)

    # Step 7: Show the images (non-blocking)
    plt.show(block=False)

    # Step 8: Loop to check for window closure
    while cv2.getWindowProperty('Original Image', 0) >= 0:
        key = cv2.waitKey(1)  # Check for key events every 1 ms

        if key == 27:  # 27 is the ASCII code for the Esc key
            print("Window closed by user.")
            break
        elif key != -1:
            print(f"Key pressed: {chr(key)}")

    # Step 9: Close the plot window
    plt.close()

# Example usage:
image_path = 'path/to/your/image.jpg'
display_images(image_path)
```

In this version, we use `cv2.getWindowProperty('Original Image', 0)` to check if the window with the title 'Original Image' is still open. If the property is less than 0, it means the window is closed, and the loop will break. This approach is more robust and allows for a flexible way to close the window by either using the close button or pressing the Esc key.




Certainly! Let's continue by making the code more modular and flexible. We'll create a function that encapsulates the logic for displaying an image and waiting for user input. This function can be reused for different images. Additionally, we'll handle the case where the window is closed using the close button in a more graceful manner:

```python
import cv2
import matplotlib.pyplot as plt

def display_image(image, window_title='Image', wait_key=True):
    # Display the image using matplotlib
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Loop to check for window closure or wait for key press
    while cv2.getWindowProperty(window_title, 0) >= 0:
        if wait_key:
            key = cv2.waitKey(1)  # Check for key events every 1 ms

            if key == 27:  # 27 is the ASCII code for the Esc key
                print("Window closed by user.")
                break
            elif key != -1:
                print(f"Key pressed: {chr(key)}")
        else:
            # Wait for a key press to close the window
            cv2.waitKey(0)
            break

    # Close the plot window
    plt.close()

# Example usage:
image_path = 'path/to/your/image.jpg'
original_image = cv2.imread(image_path)

if original_image is not None:
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Resize images for better visualization
    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    # Display the original and grayscale images
    display_image(resized_original, window_title='Original Image')
    display_image(resized_grayscale, window_title='Grayscale Image')
else:
    print(f"Error: Unable to load the image at {image_path}")
```

In this version, the logic for displaying an image and waiting for user input is encapsulated in the `display_image` function. This function can be called with different images and window titles. It handles both keyboard input and window closure using the close button more gracefully. The example usage demonstrates how to display the original and grayscale images using this function.




Certainly! Let's continue by addressing a potential issue related to closing the window with the close button. Specifically, when using `cv2.waitKey` in a non-blocking mode, it may not capture the key event for closing the window. To handle this scenario, we'll use the `cv2.getWindowProperty` method in combination with `cv2.waitKey` to efficiently check for both keyboard events and window closure:

```python
import cv2
import matplotlib.pyplot as plt

def display_image(image, window_title='Image', wait_key=True):
    # Display the image using matplotlib
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Loop to check for window closure or wait for key press
    while cv2.getWindowProperty(window_title, 0) >= 0:
        if wait_key:
            key = cv2.waitKey(1) & 0xFF  # Check for key events every 1 ms

            if key == 27:  # 27 is the ASCII code for the Esc key
                print("Window closed by user.")
                break
            elif key != 255:  # 255 is returned when no key is pressed
                print(f"Key pressed: {chr(key)}")
        else:
            # Wait for a key press to close the window
            cv2.waitKey(0)
            break

    # Close the plot window
    plt.close()

# Example usage:
image_path = 'path/to/your/image.jpg'
original_image = cv2.imread(image_path)

if original_image is not None:
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Resize images for better visualization
    resized_original = cv2.resize(original_image, (800, 600))
    resized_grayscale = cv2.resize(grayscale_image, (800, 600))

    # Display the original and grayscale images
    display_image(resized_original, window_title='Original Image')
    display_image(resized_grayscale, window_title='Grayscale Image')
else:
    print(f"Error: Unable to load the image at {image_path}")
```

In this version, the `cv2.waitKey(1) & 0xFF` ensures that the key code is properly extracted, and the additional check for `key != 255` handles the case when no key is pressed. This modification enhances the reliability of capturing both keyboard events and window closure.





Certainly! Let's continue by refining the code to allow for more customization, such as specifying the window size and title. Additionally, we'll incorporate the improvements made so far into the example usage:

```python
import cv2
import matplotlib.pyplot as plt

def display_image(image, window_title='Image', window_size=(800, 600), wait_key=True):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Loop to check for window closure or wait for key press
    while cv2.getWindowProperty(window_title, 0) >= 0:
        if wait_key:
            key = cv2.waitKey(1) & 0xFF  # Check for key events every 1 ms

            if key == 27:  # 27 is the ASCII code for the Esc key
                print("Window closed by user.")
                break
            elif key != 255:  # 255 is returned when no key is pressed
                print(f"Key pressed: {chr(key)}")
        else:
            # Wait for a key press to close the window
            cv2.waitKey(0)
            break

    # Close the plot window
    plt.close()

# Example usage:
image_path = 'path/to/your/image.jpg'
original_image = cv2.imread(image_path)

if original_image is not None:
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Display the original and grayscale images with customization
    display_image(original_image, window_title='Original Image', window_size=(800, 600))
    display_image(grayscale_image, window_title='Grayscale Image', window_size=(800, 600))
else:
    print(f"Error: Unable to load the image at {image_path}")
```

In this version, the `display_image` function now accepts parameters for window title (`window_title`), window size (`window_size`), and whether to wait for a key press (`wait_key`). This makes the function more versatile and allows you to customize the display of images according to your requirements. The example usage demonstrates how to display the original and grayscale images with specific window sizes and titles.




Certainly! Let's continue by making the code more user-friendly and adding a function to prompt the user for input. Additionally, we'll modify the `display_image` function to return the key pressed by the user, providing more flexibility for further processing:

```python
import cv2
import matplotlib.pyplot as plt

def display_image(image, window_title='Image', window_size=(800, 600)):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_path = 'path/to/your/image.jpg'
original_image = cv2.imread(image_path)

if original_image is not None:
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Display the original and grayscale images with customization
    key_original = display_image(original_image, window_title='Original Image', window_size=(800, 600))
    key_grayscale = display_image(grayscale_image, window_title='Grayscale Image', window_size=(800, 600))

    # Prompt the user for input
    prompt_user()

    print(f"Key pressed for original image: {chr(key_original)}")
    print(f"Key pressed for grayscale image: {chr(key_grayscale)}")
else:
    print(f"Error: Unable to load the image at {image_path}")
```

In this version, the `display_image` function returns the key pressed by the user, and we've added a `prompt_user` function to demonstrate a simple way to interact with the user by taking input. This makes the code more modular and extensible for handling user interactions along with image display.




Certainly! Let's continue by enhancing the code to handle multiple images and their corresponding user inputs. We'll create a function that takes a list of images and displays them sequentially, allowing the user to interact with each image individually:

```python
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
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs
    user_inputs_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'])
    user_inputs_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'])

    # Prompt the user for input
    prompt_user()

    # Print user inputs for each image
    for title, key in user_inputs_original:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, key in user_inputs_grayscale:
        print(f"Key pressed for {title}: {chr(key)}")
else:
    print("Error: Unable to load one or more images.")
```

In this version, the `display_images` function takes a list of images and their corresponding window titles, displaying them one by one. The user inputs (keys pressed) for each image are collected and printed. The `prompt_user` function is used to demonstrate additional user interaction between displaying images. Adjust the `image_paths` list with the paths to your images.




Certainly! Let's further enhance the code by allowing the user to navigate between images using keyboard input. We'll modify the `display_images` function to wait for specific keys to move to the next or previous image. Additionally, we'll add a function to handle user navigation:

```python
import cv2
import matplotlib.pyplot as plt

def display_images(images, window_titles=None, window_size=(800, 600)):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    for i, (image, title) in enumerate(zip(images, window_titles)):
        key = display_image(image, window_title=title, window_size=window_size)

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, or any other key to continue.")
      
        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                continue
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 2  # Decrement by 2 to account for the automatic increment in the loop

        user_inputs.append((title, key))

    return user_inputs

def display_image(image, window_title='Image', window_size=(800, 600)):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with navigation
    user_inputs_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'])
    user_inputs_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'])

    # Prompt the user for input
    prompt_user()

    # Print user inputs for each image
    for title, key in user_inputs_original:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, key in user_inputs_grayscale:
        print(f"Key pressed for {title}: {chr(key)}")
else:
    print("Error: Unable to load one or more images.")
```

In this version, after displaying each image, the code waits for user input. If the user presses 'n', it moves to the next image (if available); if 'p', it moves to the previous image (if available). This allows the user to navigate between images using keyboard input. The code also prints the key pressed and provides instructions for navigation. Adjust the `image_paths` list with the paths to your images.




Certainly! Let's continue by improving the navigation functionality and allowing the user to exit the image display loop at any point. We'll introduce a 'q' key press to exit the loop and display a summary of user inputs after the loop ends:

```python
import cv2
import matplotlib.pyplot as plt

def display_images(images, window_titles=None, window_size=(800, 600)):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size)

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break

        user_inputs.append((title, key))

    return user_inputs

def display_image(image, window_title='Image', window_size=(800, 600)):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with improved navigation
    user_inputs_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'])
    user_inputs_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'])

    # Prompt the user for input
    prompt_user()

    # Print user inputs for each image
    for title, key in user_inputs_original:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, key in user_inputs_grayscale:
        print(f"Key pressed for {title}: {chr(key)}")
else:
    print("Error: Unable to load one or more images.")
```

In this version, the loop continues until the user presses 'q' to quit. The code now breaks out of the loop if the user presses 'q' and prints a summary of user inputs after the loop ends. This allows for more interactive and flexible navigation through the images. Adjust the `image_paths` list with the paths to your images.




Certainly! Let's continue by adding a function that allows the user to input comments for each image during the display loop. We'll also display the comments along with the user inputs after the loop ends:

```python
import cv2
import matplotlib.pyplot as plt

def display_images(images, window_titles=None, window_size=(800, 600)):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation
    comments = {}

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size)

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add a comment for the current image
            comment = input(f"Enter a comment for {title}: ")
            comments[title] = comment

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600)):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'])
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'])

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
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
```

In this version, the user can press 'c' during the display loop to add comments for the current image. The comments are stored in the `comments` dictionary, and they are printed along with the user inputs after the loop ends. This allows the user to annotate and provide comments for each displayed image. Adjust the `image_paths` list with the paths to your images.




Certainly! Let's continue by adding the ability to save user inputs and comments to a file for later reference. We'll use a CSV (Comma-Separated Values) file to store this information. Additionally, we'll create functions to load and save user inputs and comments:

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600)):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []
    comments = {}

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size)

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add a comment for the current image
            comment = input(f"Enter a comment for {title}: ")
            comments[title] = comment

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600)):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'])
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'])

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")
else:
    print("Error: Unable to load one or more images.")
```

In this version, the code introduces functions `save_to_csv` and `load_from_csv` to handle the saving and loading of user inputs and comments to/from a CSV file. The example usage demonstrates how to load existing data from the CSV file, display images

, collect new user inputs and comments, combine the new data with the loaded data, and finally, save the updated data back to the CSV file. Adjust the `image_paths` list with the paths to your images and modify the `csv_file` variable as needed.




Certainly! Let's continue by adding a feature to display the comments along with the images. We'll modify the `display_image` function to include the comments and improve the overall display of information:

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add a comment for the current image
            comment = input(f"Enter a comment for {title}: ")
            comments[title] = comment

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs,

 comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")
else:
    print("Error: Unable to load one or more images.")
```

In this version, the `display_image` function is updated to display comments along with the images. If a comment is available for the current image, it will be displayed at the top left corner of the image. Adjust the `image_paths` list with the paths to your images and modify the `csv_file` variable as needed.




Certainly! Let's further enhance the code by adding a feature to edit and update comments during the image display loop. We'll modify the `display_image` function to allow the user to edit comments interactively:

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")
else:
    print("Error: Unable to load one or more images.")
```

In this version, the `display_image` function now allows the user to add or edit comments for the current image interactively during the display loop. When the user presses 'c', they are prompted to enter or edit a comment for the image. The comments are then updated in the `comments` dictionary. Adjust the `image_paths` list with the paths to your images and modify the `csv_file` variable as needed.




Certainly! Let's continue by enhancing the code to allow the user to delete comments for the current image. We'll add a feature to delete comments during the image display loop when the user presses the 'd' key. Additionally, we'll improve the display of comments by providing more information:

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv

(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")
else:
    print("Error: Unable to load one or more images.")
```

In this version, the `display_image` function now allows the user to delete comments for the current image by pressing the 'd' key during the display loop. If a comment exists for the current image, it will be deleted and a corresponding message will be displayed. Adjust the `image_paths` list with the paths to your images and modify the `csv_file` variable as needed.




Certainly! Let's continue by adding a feature to display a summary of user inputs and comments at the end of the interactive session. This summary will include the image titles, keys pressed, and comments. Additionally, we'll provide an option for the user to save this summary to a text file for future reference:

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

def display_summary(user_inputs, comments):
    print("\nSummary of User Inputs:")
    print("------------------------")
    for title, key in user_inputs:
        comment = comments.get(title, '')
        print(f"{title}: Key pressed - {chr(key)}, Comment - {comment}")

def save_summary_to_file(user_inputs, comments, file_name='user_inputs_summary.txt'):
    with open(file_name, 'w') as file:
        file.write("Summary of User Inputs:\n")
        file.write("------------------------\n")
        for title, key in user_inputs:
            comment = comments.get(title, '')
            file.write(f"{title}: Key pressed - {chr(key)}, Comment - {comment}\n")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments

=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)
else:
    print("Error: Unable to load one or more images.")
```

In this version, the `display_summary` function is added to print a summary of user inputs and comments at the end of the interactive session. Additionally, the `save_summary_to_file` function is introduced to save this summary to a text file. Adjust the `image_paths` list with the paths to your images and modify the `csv_file` variable as needed.




Certainly! Let's further enhance the code by providing an option for the user to load an existing summary from a text file. We'll create a function called `load_summary_from_file` to load the summary from a text file and display it. Additionally, we'll improve the formatting of the displayed summary:

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

def display_summary(user_inputs, comments):
    print("\nSummary of User Inputs:")
    print("------------------------")
    for title, key in user_inputs:
        comment = comments.get(title, '')
        print(f"{title}: Key pressed - {chr(key)}, Comment - {comment}")

def save_summary_to_file(user_inputs, comments, file_name='user_inputs_summary.txt'):
    with open(file_name, 'w') as file:
        file.write("Summary of User Inputs:\n")
        file.write("------------------------\n")
        for title, key in user_inputs:
            comment = comments.get(title, '')
            file.write(f"{title}: Key pressed - {chr(key)}, Comment - {comment}\n")

def load_summary_from_file(file_name='user_inputs_summary.txt'):
    user_inputs = []
    comments = {}

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            title = None
            key = None

            for line in lines:
                line = line.strip()

                if line.startswith('Image'):
                    title = line.split(':')[0].strip()
                elif line.startswith('Key pressed'):
                    key = line.split('-')[-1].strip()
                elif line.startswith('Comment'):
                    comment = line.split('-')[-1].strip()
                    user_inputs.append((title, ord(key)))
                    comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', '

Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)
else:
    print("Error: Unable to load one or more images.")
```

In this version, the `load_summary_from_file` function is added to load an existing summary from a text file, and the example usage is updated to demonstrate loading both user inputs from a CSV file and a summary from a text file. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's continue by adding a feature to allow the user to update comments in the summary. We'll modify the `display_summary` function to interactively prompt the user for comment updates and update the summary accordingly.

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

def display_summary(user_inputs, comments):
    print("\nSummary of User Inputs:")
    print("------------------------")
    for title, key in user_inputs:
        comment = comments.get(title, '')
        print(f"{title}: Key pressed - {chr(key)}, Comment - {comment}")

def save_summary_to_file(user_inputs, comments, file_name='user_inputs_summary.txt'):
    with open(file_name, 'w') as file:
        file.write("Summary of User Inputs:\n")
        file.write("------------------------\n")
        for title, key in user_inputs:
            comment = comments.get(title, '')
            file.write(f"{title}: Key pressed - {chr(key)}, Comment - {comment}\n")

def load_summary_from_file(file_name='user_inputs_summary.txt'):
    user_inputs = []
    comments = {}

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            title = None
            key = None

            for line in lines:
                line = line.strip()

                if line.startswith('Image'):
                    title = line.split(':')[0].strip()
                elif line.startswith('Key pressed'):
                    key = line.split('-')[-1].strip()
                elif line.startswith('Comment'):
                    comment = line.split('-')[-1].strip()
                    user_inputs.append((title, ord(key)))
                    comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def update_summary_comments(user_inputs, comments):
    print("\nUpdate Comments in the Summary:")
    print("--------------------------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

    try:
        choice = int(input("\nEnter the number of the entry to update (or 0 to skip): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if choice == 0 or choice > len(user_inputs):
        print("Skipping comment update.")
        return

    selected_title, selected_key = user_inputs[choice - 1]
    new_comment = input(f"Enter/Edit a new comment for {selected_title}: ")
    comments[selected_title] = new_comment
    print(f"Comment for {selected_title} updated.")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images,

 window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)
else:
    print("Error: Unable to load one or more images.")
```

In this version, the `update_summary_comments` function is added to interactively prompt the user to update comments in the summary. The example usage is updated to demonstrate this new functionality. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's continue by adding a feature to allow the user to delete entries from both the main user inputs and the summary. We'll create functions `delete_entry` and `delete_entry_from_summary` to handle these deletions interactively.

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

def display_summary(user_inputs, comments):
    print("\nSummary of User Inputs:")
    print("------------------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

def save_summary_to_file(user_inputs, comments, file_name='user_inputs_summary.txt'):
    with open(file_name, 'w') as file:
        file.write("Summary of User Inputs:\n")
        file.write("------------------------\n")
        for title, key in user_inputs:
            comment = comments.get(title, '')
            file.write(f"{title}: Key pressed - {chr(key)}, Comment - {comment}\n")

def load_summary_from_file(file_name='user_inputs_summary.txt'):
    user_inputs = []
    comments = {}

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            title = None
            key = None

            for line in lines:
                line = line.strip()

                if line.startswith('Image'):
                    title = line.split(':')[0].strip()
                elif line.startswith('Key pressed'):
                    key = line.split('-')[-1].strip()
                elif line.startswith('Comment'):
                    comment = line.split('-')[-1].strip()
                    user_inputs.append((title, ord(key)))
                    comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def update_summary_comments(user_inputs, comments):
    print("\nUpdate Comments in the Summary:")
    print("--------------------------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

    try:
        choice = int(input("\nEnter the number of the entry to update (or 0 to skip): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if choice == 0 or choice > len(user_inputs):
        print("Skipping comment update.")
        return

    selected_title, selected_key = user_inputs[choice - 1]
    new_comment = input(f"Enter/Edit a new comment for {selected_title}: ")
    comments[selected_title] = new_comment
    print(f"Comment for {selected_title} updated.")

def delete_entry(user_inputs, comments):
    print("\nDelete Entry:")
    print("--------------")
    for i

, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

    try:
        choice = int(input("\nEnter the number of the entry to delete (or 0 to skip): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if choice == 0 or choice > len(user_inputs):
        print("Skipping entry deletion.")
        return

    deleted_entry = user_inputs.pop(choice - 1)
    deleted_title, _ = deleted_entry
    if deleted_title in comments:
        del comments[deleted_title]
    print(f"Entry for {deleted_title} deleted.")

def delete_entry_from_summary(user_inputs, comments):
    print("\nDelete Entry from Summary:")
    print("---------------------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

    try:
        choice = int(input("\nEnter the number of the entry to delete from the summary (or 0 to skip): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if choice == 0 or choice > len(user_inputs):
        print("Skipping entry deletion from the summary.")
        return

    deleted_entry = user_inputs.pop(choice - 1)
    deleted_title, _ = deleted_entry
    if deleted_title in comments:
        del comments[deleted_title]
    print(f"Entry for {deleted_title} deleted from the summary.")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)
else:
    print("Error: Unable to load one or more images.")
```

In this version, two new functions, `delete_entry` and `delete_entry_from_summary`, are added to handle the deletion of entries from the main user inputs and the summary, respectively. The example usage is updated to demonstrate these new functionalities. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's continue by adding a few more features. We'll add the ability to navigate through the summary entries and display images associated with each summary entry. Additionally, we'll include an option to clear all comments in the summary.

```python
import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")

        user_inputs.append((title, key))

    return user_inputs, comments

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

def save_to_csv(user_inputs, comments, csv_file='user_inputs.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Title', 'Key Pressed', 'Comment'])

        for title, key in user_inputs:
            comment = comments.get(title, '')
            writer.writerow([title, chr(key), comment])

def load_from_csv(csv_file='user_inputs.csv'):
    user_inputs = []
    comments = {}

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header

            for row in reader:
                title, key, comment = row
                user_inputs.append((title, ord(key)))
                comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def prompt_user():
    user_input = input("Enter something: ")
    print(f"User entered: {user_input}")

def display_summary(user_inputs, comments):
    print("\nSummary of User Inputs:")
    print("------------------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

def save_summary_to_file(user_inputs, comments, file_name='user_inputs_summary.txt'):
    with open(file_name, 'w') as file:
        file.write("Summary of User Inputs:\n")
        file.write("------------------------\n")
        for title, key in user_inputs:
            comment = comments.get(title, '')
            file.write(f"{title}: Key pressed - {chr(key)}, Comment - {comment}\n")

def load_summary_from_file(file_name='user_inputs_summary.txt'):
    user_inputs = []
    comments = {}

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            title = None
            key = None

            for line in lines:
                line = line.strip()

                if line.startswith('Image'):
                    title = line.split(':')[0].strip()
                elif line.startswith('Key pressed'):
                    key = line.split('-')[-1].strip()
                elif line.startswith('Comment'):
                    comment = line.split('-')[-1].strip()
                    user_inputs.append((title, ord(key)))
                    comments[title] = comment

    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty data

    return user_inputs, comments

def update_summary_comments(user_inputs, comments):
    print("\nUpdate Comments in the Summary:")
    print("--------------------------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

    try:
        choice = int(input("\nEnter the number of the entry to update (or 0 to skip): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if choice == 0 or choice > len(user_inputs):
        print("Skipping comment update.")
        return

    selected_title, selected_key = user_inputs[choice - 1]
    new_comment = input(f"Enter/Edit a new comment for {selected_title}: ")
    comments[selected_title] = new_comment
    print(f"Comment for {selected_title} updated.")

def delete_entry(user_inputs, comments):
    print("\nDelete Entry:")
    print("--------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

    try:
        choice = int(input("\nEnter the number of the entry to delete (or 0 to skip): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if choice == 0 or choice > len(user_inputs):
        print("Skipping entry deletion.")
        return

    deleted_entry = user_inputs.pop(choice - 1)
    deleted_title, _ = deleted_entry
    if deleted_title in comments:
        del comments

[deleted_title]
    print(f"Entry for {deleted_title} deleted.")

def delete_entry_from_summary(user_inputs, comments):
    print("\nDelete Entry from Summary:")
    print("---------------------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

    try:
        choice = int(input("\nEnter the number of the entry to delete from the summary (or 0 to skip): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if choice == 0 or choice > len(user_inputs):
        print("Skipping entry deletion from the summary.")
        return

    deleted_entry = user_inputs.pop(choice - 1)
    deleted_title, _ = deleted_entry
    if deleted_title in comments:
        del comments[deleted_title]
    print(f"Entry for {deleted_title} deleted from the summary.")

def navigate_summary(user_inputs, comments, images):
    print("\nNavigate through Summary Entries:")
    print("----------------------------------")
    for i, (title, key) in enumerate(user_inputs, start=1):
        comment = comments.get(title, '')
        print(f"{i}. {title}: Key pressed - {chr(key)}, Comment - {comment}")

    try:
        choice = int(input("\nEnter the number of the entry to display the associated image (or 0 to skip): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if choice == 0 or choice > len(user_inputs):
        print("Skipping navigation through summary entries.")
        return

    selected_title, _ = user_inputs[choice - 1]
    selected_image = next((image for image in images if selected_title.lower() in image.lower()), None)

    if selected_image is not None:
        display_image(selected_image, window_title=selected_title, comment=comments.get(selected_title, None))
    else:
        print(f"No image found for {selected_title}.")

def clear_summary_comments(comments):
    print("\nClear All Comments in the Summary:")
    print("-----------------------------------")
    confirm = input("Are you sure you want to clear all comments? (y/n): ")

    if confirm.lower() == 'y':
        comments.clear()
        print("All comments cleared from the summary.")
    else:
        print("No changes made.")

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)
else:
    print("Error: Unable to load one or more images.")
```

In this version, two new functions, `navigate_summary` and `clear_summary_comments`, are added to allow the user to navigate through summary entries and display associated images, as well as to clear all comments in the summary. The example usage is updated to demonstrate these new functionalities. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's continue by adding an option to save the current state of user inputs and comments at any point during the interaction. We'll create a function called `save_state` that allows the user to save the current data to a specified file.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")

        user_inputs.append((title, key))

    return user_inputs, comments

def save_state(user_inputs, comments, file_name='saved_state.json'):
    state_data = {
        'user_inputs': user_inputs,
        'comments': comments
    }

    with open(file_name, 'w') as file:
        json.dump(state_data, file)

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `save_state` is added to save the current state of user inputs and comments to a JSON file. The example usage is updated to include an option for the user to save the current state at any point during the interaction. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's continue by adding a feature to load a saved state from a JSON file. We'll create a function called `load_state` that allows the user to load a previously saved state.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")

        user_inputs.append((title, key))

    return user_inputs, comments

def save_state(user_inputs, comments, file_name='saved_state.json'):
    state_data = {
        'user_inputs': user_inputs,
        'comments': comments
    }

    with open(file_name, 'w') as file:
        json.dump(state_data, file)

def load_state(file_name='saved_state.json'):
    try:
        with open(file_name, 'r') as file:
            state_data = json.load(file)
            return state_data['user_inputs'], state_data['comments']
    except FileNotFoundError:
        return None

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title

, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, the code is updated to include a new function called `load_state`, which loads a saved state from a JSON file. Additionally, the `display_images` function is modified to include the option to load a saved state during the interaction. The example usage is updated to demonstrate this new functionality. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.





Certainly! Let's add a feature to display statistics about the user inputs, such as the distribution of key presses and the number of comments for each image. We'll create a function called `display_statistics` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
from collections import Counter

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")

        user_inputs.append((title, key))

    return user_inputs, comments

def display_statistics(user_inputs, comments):
    # Display distribution of key presses
    key_distribution = Counter(key for _, key in user_inputs)
    print("\nDistribution of Key Presses:")
    for key, count in key_distribution.items():
        print(f"Key '{chr(key)}': {count} times")

    # Display number of comments for each image
    comments_count = Counter(len(comment) for comment in comments.values())
    print("\nNumber of Comments for Each Image:")
    for num_comments, count in comments_count.items():
        print(f"{num_comments} comments: {count} images")

def save_state(user_inputs, comments, file_name='saved_state.json'):
    state_data = {
        'user_inputs': user_inputs,
        'comments': comments
    }

    with open(file_name, 'w') as file:
        json.dump(state_data, file)

def load_state(file_name='saved_state.json'):
    try:
        with open(file_name, 'r') as file:
            state_data = json.load(file)
            return state_data['user_inputs'], state_data['comments']
    except FileNotFoundError:
        return None

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
   

 display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `display_statistics` is added to display statistics about the user inputs, including the distribution of key presses and the number of comments for each image. The example usage is updated to include displaying statistics at the end of the interaction. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add a feature to visualize the distribution of key presses using a bar chart. We'll create a function called `visualize_key_distribution` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
from collections import Counter

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")

        user_inputs.append((title, key))

    return user_inputs, comments

def visualize_key_distribution(user_inputs):
    # Display distribution of key presses using a bar chart
    key_counts = Counter(key for _, key in user_inputs)
    keys, counts = zip(*sorted(key_counts.items()))

    plt.bar(keys, counts, color='skyblue')
    plt.xlabel('Key Pressed')
    plt.ylabel('Frequency')
    plt.title('Distribution of Key Presses')
    plt.show()

def display_statistics(user_inputs, comments):
    # Display distribution of key presses
    key_distribution = Counter(key for _, key in user_inputs)
    print("\nDistribution of Key Presses:")
    for key, count in key_distribution.items():
        print(f"Key '{chr(key)}': {count} times")

    # Display number of comments for each image
    comments_count = Counter(len(comment) for comment in comments.values())
    print("\nNumber of Comments for Each Image:")
    for num_comments, count in comments_count.items():
        print(f"{num_comments} comments: {count} images")

def save_state(user_inputs, comments, file_name='saved_state.json'):
    state_data = {
        'user_inputs': user_inputs,
        'comments': comments
    }

    with open(file_name, 'w') as file:
        json.dump(state_data, file)

def load_state(file_name='saved_state.json'):
    try:
        with open(file_name, 'r') as file:
            state_data = json.load(file)
            return state_data['user_inputs'], state_data['comments']
    except FileNotFoundError:
        return None

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs

, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `visualize_key_distribution` is added to display a bar chart showing the distribution of key presses. The example usage is updated to include visualizing the key distribution at the end of the interaction. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add the final piece to the code: a function to display the images and associated comments in a slideshow mode. The user can navigate through the images and comments in a sequential manner. We'll create a function called `slideshow` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
from collections import Counter

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)

        user_inputs.append((title, key))

    return user_inputs, comments

def slideshow(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        print(f"{title}: Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, or any other key to continue.")

        key = cv2.waitKey(0) & 0xFF

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")

    cv2.destroyAllWindows()

def visualize_key_distribution(user_inputs):
    # Display distribution of key presses using a bar chart
    key_counts = Counter(key for _, key in user_inputs)
    keys, counts = zip(*sorted(key_counts.items()))

    plt.bar(keys, counts, color='skyblue')
    plt.xlabel('Key Pressed')
    plt.ylabel('Frequency')
    plt.title('Distribution of Key Presses')
    plt.show()

def display_statistics(user_inputs, comments):
    # Display distribution of key presses
    key_distribution = Counter(key for _, key in user_inputs)
    print("\nDistribution of Key Presses:")
    for key, count in key_distribution.items():
        print(f"Key '{chr(key)}': {count} times")

    # Display number of comments for each image
    comments_count = Counter(len(comment) for comment in comments.values())
    print("\nNumber

 of Comments for Each Image:")
    for num_comments, count in comments_count.items():
        print(f"{num_comments} comments: {count} images")

def save_state(user_inputs, comments, file_name='saved_state.json'):
    state_data = {
        'user_inputs': user_inputs,
        'comments': comments
    }

    with open(file_name, 'w') as file:
        json.dump(state_data, file)

def load_state(file_name='saved_state.json'):
    try:
        with open(file_name, 'r') as file:
            state_data = json.load(file)
            return state_data['user_inputs'], state_data['comments']
    except FileNotFoundError:
        return None

def display_image(image, window_title='Image', window_size=(800, 600), comment=None):
    # Resize the image for better visualization
    resized_image = cv2.resize(image, window_size)

    # Display the image using matplotlib
    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    # Display comment, if available
    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image (non-blocking)
    plt.show(block=False)

    # Wait for a key press to close the window
    key = cv2.waitKey(0) & 0xFF

    # Close the plot window
    plt.close()

    return key

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow
    slideshow(images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function called `slideshow` is added to display the images and associated comments in a sequential manner. The user can navigate through the images and comments in a slideshow mode. The example usage is updated to include starting the slideshow at the end of the interaction. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add a feature to compare the original and grayscale images side by side in the slideshow mode. Additionally, we'll include an option to toggle between the original and grayscale views during the slideshow. We'll create a function called `slideshow_compare` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
from collections import Counter

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)

        user_inputs.append((title, key))

    return user_inputs, comments

def slideshow_compare(images, grayscale_images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    i = 0  # Initialize index for navigation
    show_grayscale = False  # Flag to toggle between original and grayscale views

    while i < len(images):
        image = images[i]
        grayscale_image = grayscale_images[i]
        title = window_titles[i]

        if show_grayscale:
            key = display_image(grayscale_image, window_title=f"{title} (Grayscale)", window_size=window_size, comment=comments.get(title, None))
        else:
            key = display_image(image, window_title=f"{title} (Original)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 't' to toggle between original and grayscale views, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 't':
            # Toggle between original and grayscale views
            show_grayscale = not show_grayscale
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)

    cv2.destroyAllWindows()

def visualize_key_distribution(user_inputs):
    # Display distribution of key presses using a bar chart
    key_counts = Counter(key for _, key in user_inputs)
    keys, counts = zip(*sorted(key_counts.items()))

    plt.bar(keys, counts, color='skyblue')
    plt.xlabel('Key Pressed')
    plt.ylabel('Frequency')
    plt.title('Distribution of Key Presses')
    plt.show()

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments

_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `slideshow_compare` is added to display the original and grayscale images side by side in a sequential manner. The user can toggle between the original and grayscale views during the slideshow. The example usage is updated to include starting the slideshow with image comparison at the end of the interaction. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add a few more features to enhance the functionality of the program. Specifically, we'll add the ability to export the comments to a CSV file, and we'll include a function to search for comments based on keywords. We'll create functions called `export_comments_to_csv` and `search_comments` for these purposes.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
from collections import Counter

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)

        user_inputs.append((title, key))

    return user_inputs, comments

def export_comments_to_csv(comments, file_name='comments_export.csv'):
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['Image Title', 'Comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for title, comment in comments.items():
            writer.writerow({'Image Title': title, 'Comment': comment})

def search_comments(comments, keyword):
    found_comments = {title: comment for title, comment in comments.items() if keyword.lower() in comment.lower()}

    if found_comments:
        print(f"\nComments containing '{keyword}':")
        for title, comment in found_comments.items():
            print(f"{title}: {comment}")
    else:
        print(f"No comments found containing '{keyword}'.")

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a

 text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, two new functions, `export_comments_to_csv` and `search_comments`, are added. The example usage is updated to include options for exporting comments to a CSV file (`'e'`) and searching for comments based on keywords (`'k'`). Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's continue by adding a feature to display a histogram of pixel intensities for both the original and grayscale images. Additionally, we'll include an option to save the histograms as images. We'll create functions called `display_histogram` and `save_histogram_image` for these purposes.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
from collections import Counter

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(image, title)

        user_inputs.append((title, key))

    return user_inputs, comments

def display_histogram(image, title):
    plt.hist(image.flatten(), bins=256, range=[0,256], color='blue', alpha=0.7)
    plt.title(f'Histogram for {title}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def save_histogram_image(image, title, file_name='histogram.png'):
    plt.hist(image.flatten(), bins=256, range=[0,256], color='blue', alpha=0.7)
    plt.title(f'Histogram for {title}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig(file_name)
    print(f'Histogram image saved as {file_name}')

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs,

 comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, two new functions, `display_histogram` and `save_histogram_image`, are added. The example usage is updated to include options for displaying the histogram (`'h'`) and saving the histogram image (`'i'`). Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add a feature to apply a simple image processing filter to the images during the slideshow. We'll create a function called `apply_filter` to perform this operation. In this example, we'll use the OpenCV function `cv2.GaussianBlur` for a Gaussian blur filter, but you can customize it based on your needs.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
from collections import Counter

def apply_filter(image, filter_type='gaussian', kernel_size=(5, 5)):
    if filter_type.lower() == 'gaussian':
        return cv2.GaussianBlur(image, kernel_size, 0)
    else:
        print(f"Filter type '{filter_type}' not recognized. Returning the original image.")
        return image

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)

        key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2

.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `apply_filter` is added to apply a filter to the images during the slideshow. The example usage is updated to include the option to apply a filter (`'f'`). Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add one more feature to allow the user to rotate the displayed images during the slideshow. We'll create a function called `rotate_image` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
from collections import Counter

def rotate_image(image, angle):
    # Get image center
    center = tuple(np.array(image.shape[1::-1]) / 2)
    # Perform rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)

        key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            key = display_image(rotated_image, window_title=f"{title} (Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display

_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `rotate_image` is added to rotate the images during the slideshow. The example usage is updated to include the option to rotate the image (`'r'`). Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add one more feature to resize the displayed images during the slideshow. We'll create a function called `resize_image` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def resize_image(image, scale_percent):
    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # Resize the image
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)

        key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            key = display_image(rotated_image, window_title=f"{title} (Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            key = display_image(resized_image, window_title=f"{title} (Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image


    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `resize_image` is added to resize the images during the slideshow. The example usage is updated to include the option to resize the image (`'z'`). Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add a feature to flip the displayed images during the slideshow. We'll create a function called `flip_image` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def flip_image(image, flip_code):
    # Flip the image
    flipped_image = cv2.flip(image, flip_code)
    return flipped_image

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)

        key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            key = display_image(rotated_image, window_title=f"{title} (Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            key = display_image(resized_image, window_title=f"{title} (Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            key = display_image(flipped_image, window_title=f"{title} (Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale


    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `flip_image` is added to flip the images during the slideshow. The example usage is updated to include the option to flip the image (`'x'`). Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add a feature to adjust the brightness and contrast of the displayed images during the slideshow. We'll create a function called `adjust_brightness_contrast` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def adjust_brightness_contrast(image, alpha, beta):
    # Adjust brightness and contrast using the formula: new_pixel_value = alpha * original_pixel_value + beta
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)

        key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            key = display_image(rotated_image, window_title=f"{title

} (Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            key = display_image(resized_image, window_title=f"{title} (Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            key = display_image(flipped_image, window_title=f"{title} (Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha = float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            key = display_image(adjusted_image, window_title=f"{title} (Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `adjust_brightness_contrast` is added to adjust the brightness and contrast of the images during the slideshow. The example usage is updated to include the option to adjust the brightness and contrast (`'b'`). Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's add a feature to apply edge detection to the displayed images during the slideshow. We'll create a function called `detect_edges` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def detect_edges(image, low_threshold, high_threshold):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Create an output image with edges highlighted in red
    output_image = np.copy(image)
    output_image[edges != 0] = [0, 0, 255]  # Set the red channel to 255

    return output_image

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)

        key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            key = display_image(filtered_image, window_title=f"{title} (Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle

)
            key = display_image(rotated_image, window_title=f"{title} (Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            key = display_image(resized_image, window_title=f"{title} (Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            key = display_image(flipped_image, window_title=f"{title} (Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha = float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            key = display_image(adjusted_image, window_title=f"{title} (Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")
        elif chr(key).lower() == 'd':
            # Detect edges
            low_threshold = int(input("Enter the low threshold for edge detection: "))
            high_threshold = int(input("Enter the high threshold for edge detection: "))
            edges_image = detect_edges(original_image, low_threshold, high_threshold)
            key = display_image(edges_image, window_title=f"{title} (Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected with thresholds {low_threshold} and {high_threshold}.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `detect_edges` is added to apply edge detection to the images during the slideshow. The example usage is updated to include the option to detect edges (`'d'`). Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.




Certainly! Let's enhance the functionality by adding an option to display the original and processed images side by side during the slideshow. We'll create a function called `display_images_side_by_side` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def display_images_side_by_side(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600), comment=None):
    # Concatenate the original and processed images side by side
    side_by_side_image = np.concatenate((original_image, processed_image), axis=1)

    # Display the side-by-side image
    key = display_image(side_by_side_image, window_title=window_title, window_size=window_size, comment=comment)

    return key

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)

        key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            key = display_images_side_by_side(original_image, rotated_image, window_title=f"{title} (Original vs Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            key = display_images_side_by_side(original_image, resized_image, window_title=f"{title} (Original vs Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            key = display_images_side_by_side(original_image, flipped_image, window_title=f"{title} (Original vs Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha = float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            key = display_images_side_by_side(original_image, adjusted_image, window_title=f"{title} (Original vs Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")
        elif chr(key).lower() == 'd':
            # Detect edges
            low_threshold = int(input("Enter the low threshold for edge detection: "))
            high_threshold = int(input("Enter

 the high threshold for edge detection: "))
            edges_image = detect_edges(original_image, low_threshold, high_threshold)
            key = display_images_side_by_side(original_image, edges_image, window_title=f"{title} (Original vs Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected with thresholds {low_threshold} and {high_threshold}.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `display_images_side_by_side` is added to display the original and processed images side by side during the slideshow. The example usage is updated to include this option. Adjust the `image_paths`, `csv_file`, and `summary_file` variables as needed.





Certainly! Let's continue by adding a feature to save the processed images during the slideshow. We'll create a function called `save_processed_images` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def save_processed_images(processed_images, output_folder):
    # Save processed images to the specified folder
    for i, image in enumerate(processed_images):
        output_path = f"{output_folder}/processed_image_{i + 1}.png"
        cv2.imwrite(output_path, image)
        print(f"Processed image {i + 1} saved to {output_path}.")

def display_images_side_by_side(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600), comment=None):
    # Concatenate the original and processed images side by side
    side_by_side_image = np.concatenate((original_image, processed_image), axis=1)

    # Display the side-by-side image
    key = display_image(side_by_side_image, window_title=window_title, window_size=window_size, comment=comment)

    return key

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []
    processed_images = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)
        processed_images.append(filtered_image)

        key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            processed_images[-1] = filtered_image
            key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            processed_images[-1] = rotated_image
            key = display_images_side_by_side(original_image, rotated_image, window_title=f"{title} (Original vs Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            processed_images[-1] = resized_image
            key = display_images_side_by_side(original_image, resized_image, window_title=f"{title} (Original vs Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            processed_images[-1] = flipped_image
            key = display_images_side_by_side(original_image, flipped_image, window_title=f"{title} (Original vs Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha =

 float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            processed_images[-1] = adjusted_image
            key = display_images_side_by_side(original_image, adjusted_image, window_title=f"{title} (Original vs Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")
        elif chr(key).lower() == 'd':
            # Detect edges
            low_threshold = int(input("Enter the low threshold for edge detection: "))
            high_threshold = int(input("Enter the high threshold for edge detection: "))
            edges_image = detect_edges(original_image, low_threshold, high_threshold)
            processed_images[-1] = edges_image
            key = display_images_side_by_side(original_image, edges_image, window_title=f"{title} (Original vs Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected with thresholds {low_threshold} and {high_threshold}.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if processed_images:
                processed_images.pop()
                print("Last action undone.")
            else:
                print("No action to undo.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'
output_folder = 'processed_images'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Save processed images
    save_processed_images(processed_images, output_folder)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, a new function `save_processed_images` is added to save the processed images to a specified folder. The example usage is updated to include this option. Adjust the `image_paths`, `csv_file`, `summary_file`, and `output_folder` variables as needed.





Certainly! Let's continue by incorporating a feature to view the history of processed images during the slideshow. We'll create a function called `display_image_history` to show the history of processed images for a specific image.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def display_image_history(title, image_history, window_title="Image History", window_size=(800, 600)):
    plt.figure(figsize=(12, 6))

    for i, image in enumerate(image_history, 1):
        plt.subplot(1, len(image_history), i)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Step {i}")

    plt.suptitle(f"Processing History for {title}")
    plt.tight_layout()
    plt.show()

# ... (rest of the code remains the same)

def display_images_side_by_side(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600), comment=None):
    # Concatenate the original and processed images side by side
    side_by_side_image = np.concatenate((original_image, processed_image), axis=1)

    # Display the side-by-side image
    key = display_image(side_by_side_image, window_title=window_title, window_size=window_size, comment=comment)

    return key

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []
    processed_images = []
    image_history = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)
        processed_images.append(filtered_image)
        image_history.append(filtered_image.copy())

        key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, 'h' to view image history, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
                processed_images = []
                image_history = []
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
                processed_images = []
                image_history = []
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            processed_images.append(filtered_image)
            image_history.append(filtered_image.copy())
            key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            #

 Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            processed_images.append(rotated_image)
            image_history.append(rotated_image.copy())
            key = display_images_side_by_side(original_image, rotated_image, window_title=f"{title} (Original vs Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            processed_images.append(resized_image)
            image_history.append(resized_image.copy())
            key = display_images_side_by_side(original_image, resized_image, window_title=f"{title} (Original vs Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            processed_images.append(flipped_image)
            image_history.append(flipped_image.copy())
            key = display_images_side_by_side(original_image, flipped_image, window_title=f"{title} (Original vs Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha = float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            processed_images.append(adjusted_image)
            image_history.append(adjusted_image.copy())
            key = display_images_side_by_side(original_image, adjusted_image, window_title=f"{title} (Original vs Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")
        elif chr(key).lower() == 'd':
            # Detect edges
            low_threshold = int(input("Enter the low threshold for edge detection: "))
            high_threshold = int(input("Enter the high threshold for edge detection: "))
            edges_image = detect_edges(original_image, low_threshold, high_threshold)
            processed_images.append(edges_image)
            image_history.append(edges_image.copy())
            key = display_images_side_by_side(original_image, edges_image, window_title=f"{title} (Original vs Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected with thresholds {low_threshold} and {high_threshold}.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if processed_images:
                processed_images.pop()
                image_history.pop()
                print("Last action undone.")
            else:
                print("No action to undo.")
        elif chr(key).lower() == 't':
            # Save processed images
            save_processed_images(processed_images, 'processed_images')
            print("Processed images saved.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete
```





Certainly! Let's continue by adding a feature to display image information during the slideshow, such as dimensions and color channels. We'll create a function called `display_image_info` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def display_image_info(image, window_title="Image Information"):
    # Get image dimensions and number of color channels
    height, width, channels = image.shape
    info_text = f"Dimensions: {width} x {height}\nColor Channels: {channels}"

    # Display the image information
    cv2.imshow(window_title, np.zeros((1, 1), dtype=np.uint8))  # Create an empty window
    cv2.waitKey(1)  # Wait for a moment to create the window
    cv2.destroyAllWindows()  # Close the empty window

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 400, 100)
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ... (previous code remains the same)

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []
    processed_images = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)
        processed_images.append(filtered_image)

        key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Display image information
        display_image_info(original_image, window_title=f"{title} Information")

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            processed_images[-1] = filtered_image
            key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image

(original_image, angle)
            processed_images[-1] = rotated_image
            key = display_images_side_by_side(original_image, rotated_image, window_title=f"{title} (Original vs Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            processed_images[-1] = resized_image
            key = display_images_side_by_side(original_image, resized_image, window_title=f"{title} (Original vs Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            processed_images[-1] = flipped_image
            key = display_images_side_by_side(original_image, flipped_image, window_title=f"{title} (Original vs Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha = float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            processed_images[-1] = adjusted_image
            key = display_images_side_by_side(original_image, adjusted_image, window_title=f"{title} (Original vs Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")
        elif chr(key).lower() == 'd':
            # Detect edges
            low_threshold = int(input("Enter the low threshold for edge detection: "))
            high_threshold = int(input("Enter the high threshold for edge detection: "))
            edges_image = detect_edges(original_image, low_threshold, high_threshold)
            processed_images[-1] = edges_image
            key = display_images_side_by_side(original_image, edges_image, window_title=f"{title} (Original vs Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected with thresholds {low_threshold} and {high_threshold}.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if processed_images:
                processed_images.pop()
                print("Last action undone.")
            else:
                print("No action to undo.")
        elif chr(key).lower() == 't':
            # Save processed images
            save_processed_images(processed_images, 'processed_images')
            print("Processed images saved to 'processed_images' folder.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'
output_folder = 'processed_images'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in
```




Certainly! Let's continue by adding a feature to compare the processed images during the slideshow. We'll create a function called `slideshow_compare` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def save_processed_images(processed_images, output_folder):
    # Save processed images to the specified folder
    for i, image in enumerate(processed_images):
        output_path = f"{output_folder}/processed_image_{i + 1}.png"
        cv2.imwrite(output_path, image)
        print(f"Processed image {i + 1} saved to {output_path}.")

def display_images_side_by_side(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600), comment=None):
    # Concatenate the original and processed images side by side
    side_by_side_image = np.concatenate((original_image, processed_image), axis=1)

    # Display the side-by-side image
    key = display_image(side_by_side_image, window_title=window_title, window_size=window_size, comment=comment)

    return key

def slideshow_compare(original_images, processed_images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []

    i = 0  # Initialize index for navigation

    while i < len(original_images):
        original_image = original_images[i]
        processed_image = processed_images[i]
        title = window_titles[i]

        key = display_images_side_by_side(original_image, processed_image, window_title=f"{title} (Original vs Processed)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(original_images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            key = display_images_side_by_side(original_image, rotated_image, window_title=f"{title} (Original vs Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%

): "))
            resized_image = resize_image(original_image, scale_percent)
            key = display_images_side_by_side(original_image, resized_image, window_title=f"{title} (Original vs Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            key = display_images_side_by_side(original_image, flipped_image, window_title=f"{title} (Original vs Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha = float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            key = display_images_side_by_side(original_image, adjusted_image, window_title=f"{title} (Original vs Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")
        elif chr(key).lower() == 'd':
            # Detect edges
            low_threshold = int(input("Enter the low threshold for edge detection: "))
            high_threshold = int(input("Enter the high threshold for edge detection: "))
            edges_image = detect_edges(original_image, low_threshold, high_threshold)
            key = display_images_side_by_side(original_image, edges_image, window_title=f"{title} (Original vs Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected with thresholds {low_threshold} and {high_threshold}.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if processed_images:
                processed_images.pop()
                print("Last action undone.")
            else:
                print("No action to undo.")
        elif chr(key).lower() == 't':
            # Save processed images
            output_folder = input("Enter the folder path to save processed images: ")
            save_processed_images(processed_images, output_folder)

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'
output_folder = 'processed_images'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Save processed images
    save_processed_images(processed_images, output_folder)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this





Certainly! Let's add the ability to display processed images during the slideshow and provide options to adjust the processing parameters interactively. Additionally, we'll create a function to undo the last action.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def save_processed_images(processed_images, output_folder):
    # Save processed images to the specified folder
    for i, image in enumerate(processed_images):
        output_path = f"{output_folder}/processed_image_{i + 1}.png"
        cv2.imwrite(output_path, image)
        print(f"Processed image {i + 1} saved to {output_path}.")

def undo_last_action(processed_images, original_images, window_titles, window_size, comments):
    # Undo the last action
    if processed_images:
        processed_images.pop()
        title = window_titles[-1]
        original_image = original_images[-1]
        processed_image = processed_images[-1] if processed_images else original_image
        key = display_images_side_by_side(original_image, processed_image, window_title=f"{title} (Original vs Processed)", window_size=window_size, comment=comments.get(title, None))
        print("Last action undone.")
    else:
        print("No action to undo.")

def display_images_side_by_side(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600), comment=None):
    # Concatenate the original and processed images side by side
    side_by_side_image = np.concatenate((original_image, processed_image), axis=1)

    # Display the side-by-side image
    key = display_image(side_by_side_image, window_title=window_title, window_size=window_size, comment=comment)

    return key

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []
    processed_images = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)
        processed_images.append(filtered_image)

        key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            processed_images[-1] = filtered_image
            key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window

_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            processed_images[-1] = rotated_image
            key = display_images_side_by_side(original_image, rotated_image, window_title=f"{title} (Original vs Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            processed_images[-1] = resized_image
            key = display_images_side_by_side(original_image, resized_image, window_title=f"{title} (Original vs Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            processed_images[-1] = flipped_image
            key = display_images_side_by_side(original_image, flipped_image, window_title=f"{title} (Original vs Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha = float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            processed_images[-1] = adjusted_image
            key = display_images_side_by_side(original_image, adjusted_image, window_title=f"{title} (Original vs Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")
        elif chr(key).lower() == 'd':
            # Detect edges
            low_threshold = int(input("Enter the low threshold for edge detection: "))
            high_threshold = int(input("Enter the high threshold for edge detection: "))
            edges_image = detect_edges(original_image, low_threshold, high_threshold)
            processed_images[-1] = edges_image
            key = display_images_side_by_side(original_image, edges_image, window_title=f"{title} (Original vs Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected with thresholds {low_threshold} and {high_threshold}.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            undo_last_action(processed_images, images, window_titles, window_size, comments)
        elif chr(key).lower() == 't':
            # Save processed images
            output_folder = input("Enter the output folder for processed images: ")
            save_processed_images(processed_images, output_folder)
            print("Processed images saved.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
```





Certainly! Let's continue by adding a feature to apply a custom user-defined filter during the slideshow. We'll create a function called `apply_custom_filter` for this purpose.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def apply_custom_filter(image, kernel):
    # Apply a custom user-defined filter to the image
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def display_images_side_by_side(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600), comment=None):
    # Concatenate the original and processed images side by side
    side_by_side_image = np.concatenate((original_image, processed_image), axis=1)

    # Display the side-by-side image
    key = display_image(side_by_side_image, window_title=window_title, window_size=window_size, comment=comment)

    return key

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []
    processed_images = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)
        processed_images.append(filtered_image)

        key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, 'y' to apply custom filter, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filter_type = input("Enter the filter type (e.g., 'gaussian'): ")
            filtered_image = apply_filter(original_image, filter_type=filter_type)
            processed_images[-1] = filtered_image
            key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title} with type '{filter_type}'.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            angle = int(input("Enter the rotation angle (in degrees): "))
            rotated_image = rotate_image(original_image, angle)
            processed_images[-1] = rotated_image
            key = display_images_side_by_side(original_image, rotated_image, window_title=f"{title} (Original vs Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated by {angle} degrees.")
        elif chr(key).lower() == 'z':
            # Resize the image
            scale_percent = int(input("Enter the resize

 scale percentage (e.g., 50 for 50%): "))
            resized_image = resize_image(original_image, scale_percent)
            processed_images[-1] = resized_image
            key = display_images_side_by_side(original_image, resized_image, window_title=f"{title} (Original vs Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized by {scale_percent}%.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
            flipped_image = flip_image(original_image, flip_code)
            processed_images[-1] = flipped_image
            key = display_images_side_by_side(original_image, flipped_image, window_title=f"{title} (Original vs Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped with code {flip_code}.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            alpha = float(input("Enter the alpha value (contrast): "))
            beta = int(input("Enter the beta value (brightness): "))
            adjusted_image = adjust_brightness_contrast(original_image, alpha, beta)
            processed_images[-1] = adjusted_image
            key = display_images_side_by_side(original_image, adjusted_image, window_title=f"{title} (Original vs Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted with alpha={alpha} and beta={beta}.")
        elif chr(key).lower() == 'd':
            # Detect edges
            low_threshold = int(input("Enter the low threshold for edge detection: "))
            high_threshold = int(input("Enter the high threshold for edge detection: "))
            edges_image = detect_edges(original_image, low_threshold, high_threshold)
            processed_images[-1] = edges_image
            key = display_images_side_by_side(original_image, edges_image, window_title=f"{title} (Original vs Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected with thresholds {low_threshold} and {high_threshold}.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if processed_images:
                processed_images.pop()
                print("Last action undone.")
            else:
                print("No action to undo.")
        elif chr(key).lower() == 't':
            # Save processed images
            save_processed_images(processed_images, 'processed_images')
            print("Processed images saved.")
        elif chr(key).lower() == 'y':
            # Apply a custom filter
            custom_filter_str = input("Enter the custom filter kernel (comma-separated values): ")
            kernel = np.array([list(map(int, row.split(','))) for row in custom_filter_str.split(',')])
            custom_filtered_image = apply_custom_filter(original_image, kernel)
            processed_images[-1] = custom_filtered_image
            key = display_images_side_by_side(original_image, custom_filtered_image, window_title=f"{title} (Original vs Custom Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} filtered with custom kernel: {kernel}.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary
```





Certainly! Let's continue by adding the ability to adjust the parameters of the image processing operations during the slideshow. We'll create functions for interactive parameter adjustment for the filter, rotation, resizing, flipping, brightness, contrast, and edge detection.

```python
import cv2
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import Counter

def adjust_parameters_for_filter(original_image, filter_type=None):
    if filter_type is None:
        filter_type = input("Enter the filter type (e.g., 'gaussian'): ")

    # Get additional parameters based on the filter type
    if filter_type.lower() == 'gaussian':
        sigma = float(input("Enter the sigma value for Gaussian filter: "))
        return apply_filter(original_image, filter_type=filter_type, sigma=sigma)
    elif filter_type.lower() == 'median':
        kernel_size = int(input("Enter the kernel size for Median filter: "))
        return apply_filter(original_image, filter_type=filter_type, kernel_size=kernel_size)
    else:
        return apply_filter(original_image, filter_type=filter_type)

def adjust_parameters_for_rotation(original_image, angle=None):
    if angle is None:
        angle = int(input("Enter the rotation angle (in degrees): "))
    return rotate_image(original_image, angle)

def adjust_parameters_for_resize(original_image, scale_percent=None):
    if scale_percent is None:
        scale_percent = int(input("Enter the resize scale percentage (e.g., 50 for 50%): "))
    return resize_image(original_image, scale_percent)

def adjust_parameters_for_flip(original_image, flip_code=None):
    if flip_code is None:
        flip_code = int(input("Enter the flip code (0 for horizontal, 1 for vertical, 2 for both): "))
    return flip_image(original_image, flip_code)

def adjust_parameters_for_brightness_contrast(original_image, alpha=None, beta=None):
    if alpha is None:
        alpha = float(input("Enter the alpha value (contrast): "))
    if beta is None:
        beta = int(input("Enter the beta value (brightness): "))
    return adjust_brightness_contrast(original_image, alpha, beta)

def adjust_parameters_for_edge_detection(original_image, low_threshold=None, high_threshold=None):
    if low_threshold is None:
        low_threshold = int(input("Enter the low threshold for edge detection: "))
    if high_threshold is None:
        high_threshold = int(input("Enter the high threshold for edge detection: "))
    return detect_edges(original_image, low_threshold, high_threshold)

def display_images_side_by_side(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600), comment=None):
    # Concatenate the original and processed images side by side
    side_by_side_image = np.concatenate((original_image, processed_image), axis=1)

    # Display the side-by-side image
    key = display_image(side_by_side_image, window_title=window_title, window_size=window_size, comment=comment)

    return key

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []
    processed_images = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)
        processed_images.append(filtered_image)

        key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        elif chr(key).lower() == 'c':
            # Add or edit a comment for the current image
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            # Delete the comment for the current image
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            # Save the current state
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
            # Load a saved state
            loaded_state = load_state('saved_state.json')
            if loaded_state:
                user_inputs, comments = loaded_state
                print("Saved state loaded.")
            else:
                print("No saved state found.")
        elif chr

(key).lower() == 'v':
            # Visualize key distribution
            visualize_key_distribution(user_inputs)
        elif chr(key).lower() == 'e':
            # Export comments to CSV
            export_comments_to_csv(comments)
            print("Comments exported to 'comments_export.csv'.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keyword = input("Enter a keyword to search for comments: ")
            search_comments(comments, search_keyword)
        elif chr(key).lower() == 'h':
            # Display histogram of pixel intensities
            display_histogram(original_image, title)
        elif chr(key).lower() == 'i':
            # Save histogram image
            save_histogram_image(original_image, title)
        elif chr(key).lower() == 'f':
            # Apply a filter to the image
            filtered_image = adjust_parameters_for_filter(original_image)
            processed_images[-1] = filtered_image
            key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))
            print(f"Filter applied to {title}.")
        elif chr(key).lower() == 'r':
            # Rotate the image
            rotated_image = adjust_parameters_for_rotation(original_image)
            processed_images[-1] = rotated_image
            key = display_images_side_by_side(original_image, rotated_image, window_title=f"{title} (Original vs Rotated)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} rotated.")
        elif chr(key).lower() == 'z':
            # Resize the image
            resized_image = adjust_parameters_for_resize(original_image)
            processed_images[-1] = resized_image
            key = display_images_side_by_side(original_image, resized_image, window_title=f"{title} (Original vs Resized)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} resized.")
        elif chr(key).lower() == 'x':
            # Flip the image
            flipped_image = adjust_parameters_for_flip(original_image)
            processed_images[-1] = flipped_image
            key = display_images_side_by_side(original_image, flipped_image, window_title=f"{title} (Original vs Flipped)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} flipped.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            adjusted_image = adjust_parameters_for_brightness_contrast(original_image)
            processed_images[-1] = adjusted_image
            key = display_images_side_by_side(original_image, adjusted_image, window_title=f"{title} (Original vs Adjusted)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges
            edges_image = adjust_parameters_for_edge_detection(original_image)
            processed_images[-1] = edges_image
            key = display_images_side_by_side(original_image, edges_image, window_title=f"{title} (Original vs Edges)", window_size=window_size, comment=comments.get(title, None))
            print(f"{title} edges detected.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if processed_images:
                processed_images.pop()
                print("Last action undone.")
            else:
                print("No action to undo.")
        elif chr(key).lower() == 't':
            # Save processed images
            save_processed_images(processed_images, 'processed_images_folder')
            print("Processed images saved.")

        user_inputs.append((title, key))

    return user_inputs, comments

# ... (rest of the code remains the same)

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments
```






Certainly! Let's continue by adding a function to compare the pixel-wise differences between the original and processed images and display the result. We'll call this function `compare_images`:

```python
def compare_images(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600)):
    # Compute pixel-wise absolute differences between the images
    diff_image = cv2.absdiff(original_image, processed_image)

    # Display the original, processed, and difference images side by side
    comparison_image = np.concatenate((original_image, processed_image, diff_image), axis=1)

    # Display the comparison image
    key = display_image(comparison_image, window_title=window_title, window_size=window_size)

    return key

# Add this function to your existing code
# ...

# Modify the display_images_side_by_side function to use compare_images
def display_images_side_by_side(original_image, processed_image, window_title="Image Comparison", window_size=(800, 600), comment=None):
    # Display the original and processed images side by side
    key = compare_images(original_image, processed_image, window_title=window_title, window_size=window_size)

    return key

# ...

# Update the display_images function to use display_images_side_by_side
def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []
    processed_images = []

    i = 0  # Initialize index for navigation

    while i < len(images):
        original_image = images[i]
        title = window_titles[i]

        # Apply a filter to the image (e.g., Gaussian blur)
        filtered_image = apply_filter(original_image)
        processed_images.append(filtered_image)

        key = display_images_side_by_side(original_image, filtered_image, window_title=f"{title} (Original vs Filtered)", window_size=window_size, comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            # Move to the previous image
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            # Quit the loop
            break
        # ... (other key handling remains the same)

    return user_inputs, comments

# ...

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'
output_folder = 'processed_images'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Save processed images
    save_processed_images(processed_images, output_folder)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the

 updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, grayscale_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, we added the `compare_images` function to compare the pixel-wise differences between the original and processed images. The `display_images_side_by_side` function is updated to use `compare_images` for displaying the comparison image. The example usage is also updated to include this functionality. Adjust the `image_paths`, `csv_file`, `summary_file`, and `output_folder` variables as needed.





Certainly! Let's continue by adding the functionality to create and display an image slideshow with a comparison of original and processed images. We'll create a function called `slideshow_compare` for this purpose.

```python
import time

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        # ... (other key handling remains the same)

        time.sleep(interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, you can update the example usage section to include the slideshow functionality:

```python
# ...

# Example usage:
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'
output_folder = 'processed_images'

# Load existing data from the CSV file, if available
loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

# Load an existing summary from a text file, if available
loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    # Convert images to grayscale
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Display the images and collect user inputs with comments
    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    # Combine new inputs with loaded data
    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    # Save the updated data to the CSV file
    save_to_csv(user_inputs, comments, csv_file)

    # Save processed images
    save_processed_images(processed_images, output_folder)

    # Prompt the user for input
    prompt_user()

    # Print user inputs and comments for each image
    for title, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    # Display a summary of user inputs
    display_summary(user_inputs, comments)

    # Save the summary to a text file
    save_summary_to_file(user_inputs, comments)

    # Display an existing summary from a text file, if available
    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    # Update comments in the summary
    update_summary_comments(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry
    delete_entry(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Delete an entry from the summary
    delete_entry_from_summary(user_inputs, comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Navigate through summary entries and display associated images
    navigate_summary(user_inputs, comments, images)

    # Clear all comments in the summary
    clear_summary_comments(comments)

    # Display the updated summary
    display_summary(user_inputs, comments)

    # Display statistics
    display_statistics(user_inputs, comments)

    # Visualize key distribution
    visualize_key_distribution(user_inputs)

    # Start slideshow with image comparison
    slideshow_compare(images, processed_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments)

    # Save the final state
    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")
```

In this version, the `slideshow_compare` function is added to create an interactive slideshow for comparing original and processed images. The example usage is updated to include this functionality. Adjust the `image_paths`, `csv_file`, `summary_file`, `output_folder`, and other variables as needed.





Certainly! Let's add a feature to adjust the interval between slides during the slideshow. We'll modify the `slideshow_compare` function to accept the interval as a parameter.

```python
import time

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        # ... (other key handling remains the same)

        time.sleep(interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, when calling the `slideshow_compare` function, you can specify the interval in seconds:

```python
# Start slideshow with image comparison and a 5-second interval
slideshow_compare(images, processed_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments, interval=5)
```

This modification allows you to control the time interval between slides during the slideshow. Adjust the interval value as needed.





Certainly! Let's enhance the slideshow functionality by adding the ability to adjust the interval dynamically during the slideshow. We'll update the `slideshow_compare` function to handle user input for changing the interval on-the-fly.

```python
import time

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press '+' to increase the interval or '-' to decrease the interval on-the-fly. This allows for dynamic adjustment of the slideshow speed. You can set the `default_interval` parameter to the initial interval value when calling the function.

```python
# Start slideshow with image comparison and a default 3-second interval
slideshow_compare(images, processed_images, window_titles=['Image 1', 'Image 2', 'Image 3'], comments=comments, default_interval=3)
```

Feel free to adjust the default interval value as needed.




Certainly! Let's add a feature to rotate images during the slideshow. We'll introduce a new key, 'r', to rotate the displayed image, and we'll update the `slideshow_compare` function to handle this rotation.

```python
import time

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'r' to rotate the displayed image by 90 degrees clockwise. This adds a dynamic image manipulation feature to the slideshow.

Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value when calling the function according to your preferences.




Certainly! Let's enhance the slideshow functionality by adding the ability to resize images during the slideshow. We'll introduce a new key, 'z', to resize the displayed image, and we'll update the `slideshow_compare` function to handle this resizing.

```python
import time

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip image, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            print(f"{title}: Image resized by factor {resize_factor}.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'z' to resize the displayed image interactively. The function will prompt you to enter the resize factor, allowing you to dynamically adjust the size of the image.

Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value when calling the function according to your preferences.




Certainly! Let's continue by adding the ability to flip images horizontally or vertically during the slideshow. We'll introduce two new keys, 'x' for horizontal flip and 'y' for vertical flip, and we'll update the `slideshow_compare` function to handle these flips.

```python
import time

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            print(f"{title}: Image flipped vertically.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'x' to flip the displayed image horizontally or 'y' to flip it vertically. This adds interactive horizontal and vertical flipping functionality to the slideshow.

Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value when calling the function according to your preferences.




Certainly! Let's continue by adding the ability to adjust the brightness and contrast of images during the slideshow. We'll introduce two new keys, 'b' for adjusting brightness and contrast, and we'll update the `slideshow_compare` function to handle these adjustments.

```python
import time

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            print(f"{title}: Image brightness and contrast adjusted.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'b' to interactively adjust the brightness and contrast of the displayed image. The function will prompt you to enter the adjustment factors for brightness and contrast.

Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value when calling the function according to your preferences.




Certainly! Let's continue by adding the ability to detect edges in images during the slideshow. We'll introduce a new key, 'd', to detect edges, and we'll update the `slideshow_compare` function to handle this edge detection.

```python
import time
import cv2

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            print(f"{title}: Edges detected in the image.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'd' to detect edges in the displayed image. The function will apply the Canny edge detection algorithm to the image.

Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value when calling the function according to your preferences.




Certainly! Let's continue by adding an "undo" feature during the slideshow. We'll introduce a new key, 'u', to undo the last action, and we'll update the `slideshow_compare` function to handle this undo functionality.

```python
import time
import cv2
import numpy as np

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
               
```





Certainly! Let's continue with the 'u' key to undo the last action in the `slideshow_compare` function. If the last action was edge detection ('d'), we'll revert the image to the previous processed state.

```python
import time
import cv2
import numpy as np

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed

_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                print(f"{title}: Last action undone.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'u' to undo the last action. The function will undo the most recent image processing action, such as rotation, resizing, flipping, brightness and contrast adjustment, or edge detection.

Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value when calling the function according to your preferences.




Certainly! Let's continue by adding the functionality to save the processed images during the slideshow. We'll introduce a new key, 't', to save the processed images, and we'll update the `slideshow_compare` function to handle this saving functionality.

```python
import time
import cv2
import numpy as np
import os

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                 

 processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 't' to save the currently displayed processed image. The image will be saved in the specified `save_directory` with a filename based on the original image title.

Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value and the `save_directory` parameter when calling the function according to your preferences.




Certainly! Let's continue by adding the functionality to visualize the key distribution during the slideshow. We'll introduce a new key, 'v', to trigger the visualization, and we'll update the `slideshow_compare` function to handle this visualization.

For this purpose, we'll use OpenCV to create a simple visualization of the key distribution.

```python
import time
import cv2
import numpy as np
import os

def visualize_key_distribution(user_inputs):
    key_distribution = {key: 0 for key in set(user_inputs)}
  
    for key in user_inputs:
        key_distribution[key] += 1
  
    max_count = max(key_distribution.values())
    height = 400
    width = 600
    bar_width = width // len(key_distribution)
    bar_color = (255, 0, 0)

    distribution_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    for idx, (key, count) in enumerate(key_distribution.items()):
        bar_height = int((count / max_count) * height)
        start_point = (idx * bar_width, height - bar_height)
        end_point = ((idx + 1) * bar_width - 1, height - 1)
        cv2.rectangle(distribution_image, start_point, end_point, bar_color, thickness=cv2.FILLED)

    return distribution_image

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Record user input
        user_inputs.append(chr(key))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                 

 processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            distribution_image = visualize_key_distribution(user_inputs)
            cv2.imshow("Key Distribution", distribution_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'v' to visualize the key distribution. The distribution will be displayed as a simple bar chart, and you can press any key to close the visualization window.

Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value and the `save_directory` parameter when calling the function according to your preferences.





Certainly! Let's continue by adding the functionality to export the recorded user inputs to a CSV file. We'll introduce a new key, 'e', to trigger the export, and we'll update the `slideshow_compare` function to handle this export functionality.

```python
import time
import cv2
import numpy as np
import os
import csv

def export_user_inputs(user_inputs, export_filename="user_inputs.csv"):
    with open(export_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Index', 'Key'])
        for idx, key in enumerate(user_inputs):
            csv_writer.writerow([idx + 1, key])

def visualize_key_distribution(user_inputs):
    # ... (previous implementation remains the same)

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Record user input
        user_inputs.append(chr(key))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[

processed_image_index] = original_images[processed_image_index].copy()
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            distribution_image = visualize_key_distribution(user_inputs)
            cv2.imshow("Key Distribution", distribution_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(key).lower() == 'e':
            # Export user inputs to CSV
            export_filename = input("Enter the export filename (default: user_inputs.csv): ").strip() or "user_inputs.csv"
            export_user_inputs(user_inputs, export_filename)
            print(f"User inputs exported to {export_filename}.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'e' to export the recorded user inputs to a CSV file. You'll be prompted to enter the export filename, and the CSV file will be created with the specified name. Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value and the `save_directory` parameter when calling the function according to your preferences.





Certainly! Let's continue by adding the functionality to search for comments based on keywords. We'll introduce a new key, 'k', to trigger the search, and we'll update the `slideshow_compare` function to handle this search functionality.

```python
import time
import cv2
import numpy as np
import os
import csv

def search_comments(comments, keywords):
    matching_comments = []
    for title, comment in comments.items():
        if any(keyword.lower() in comment.lower() for keyword in keywords):
            matching_comments.append((title, comment))
    return matching_comments

def export_user_inputs(user_inputs, export_filename="user_inputs.csv"):
    with open(export_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Index', 'Key'])
        for idx, key in enumerate(user_inputs):
            csv_writer.writerow([idx + 1, key])

def visualize_key_distribution(user_inputs):
    # ... (previous implementation remains the same)

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Record user input
        user_inputs.append(chr(key))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images

[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            distribution_image = visualize_key_distribution(user_inputs)
            cv2.imshow("Key Distribution", distribution_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(key).lower() == 'e':
            # Export user inputs to CSV
            export_filename = input("Enter the export filename (default: user_inputs.csv): ").strip() or "user_inputs.csv"
            export_user_inputs(user_inputs, export_filename)
            print(f"User inputs exported to {export_filename}.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keywords = input("Enter search keywords (comma-separated): ").split(',')
            matching_comments = search_comments(comments, search_keywords)
            if matching_comments:
                print("Matching comments:")
                for match_title, match_comment in matching_comments:
                    print(f"{match_title}: {match_comment}")
            else:
                print("No matching comments found.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'k' to search for comments based on keywords. You'll be prompted to enter the keywords (comma-separated), and the matching comments will be displayed. Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value and the `save_directory` parameter when calling the function according to your preferences.





Certainly! Let's continue by adding the functionality to display the histogram of the processed image. We'll introduce a new key, 'h', to trigger the histogram display, and we'll update the `slideshow_compare` function to handle this functionality.

```python
import time
import cv2
import numpy as np
import os
import csv

def display_histogram(image, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def search_comments(comments, keywords):
    # ... (previous implementation remains the same)

def export_user_inputs(user_inputs, export_filename="user_inputs.csv"):
    # ... (previous implementation remains the same)

def visualize_key_distribution(user_inputs):
    # ... (previous implementation remains the same)

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Record user input
        user_inputs.append(chr(key))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action


            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            distribution_image = visualize_key_distribution(user_inputs)
            cv2.imshow("Key Distribution", distribution_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(key).lower() == 'e':
            # Export user inputs to CSV
            export_filename = input("Enter the export filename (default: user_inputs.csv): ").strip() or "user_inputs.csv"
            export_user_inputs(user_inputs, export_filename)
            print(f"User inputs exported to {export_filename}.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keywords = input("Enter search keywords (comma-separated): ").split(',')
            matching_comments = search_comments(comments, search_keywords)
            if matching_comments:
                print("Matching comments:")
                for match_title, match_comment in matching_comments:
                    print(f"{match_title}: {match_comment}")
            else:
                print("No matching comments found.")
        elif chr(key).lower() == 'h':
            # Display histogram of the processed image
            display_histogram(processed_image, title=f"{title} - Processed Image Histogram")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'h' to display the histogram of the processed image. The histogram will be shown using matplotlib, so make sure you have it installed (`pip install matplotlib`). Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value and the `save_directory` parameter when calling the function according to your preferences.





Certainly! Let's continue by adding the functionality to save the histogram image. We'll introduce a new key, 'i', to trigger the saving of the histogram image, and we'll update the `slideshow_compare` function to handle this functionality.

```python
import time
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def display_histogram(image, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def save_histogram_image(image, save_path, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def search_comments(comments, keywords):
    # ... (previous implementation remains the same)

def export_user_inputs(user_inputs, export_filename="user_inputs.csv"):
    # ... (previous implementation remains the same)

def visualize_key_distribution(user_inputs):
    # ... (previous implementation remains the same)

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Record user input
        user_inputs.append(chr(key))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == '

x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            distribution_image = visualize_key_distribution(user_inputs)
            cv2.imshow("Key Distribution", distribution_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(key).lower() == 'e':
            # Export user inputs to CSV
            export_filename = input("Enter the export filename (default: user_inputs.csv): ").strip() or "user_inputs.csv"
            export_user_inputs(user_inputs, export_filename)
            print(f"User inputs exported to {export_filename}.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keywords = input("Enter search keywords (comma-separated): ").split(',')
            matching_comments = search_comments(comments, search_keywords)
            if matching_comments:
                print("Matching comments:")
                for match_title, match_comment in matching_comments:
                    print(f"{match_title}: {match_comment}")
            else:
                print("No matching comments found.")
        elif chr(key).lower() == 'h':
            # Display histogram of the processed image
            display_histogram(processed_image, title=f"{title} - Processed Image Histogram")
        elif chr(key).lower() == 'i':
            # Save histogram image
            histogram_save_path = os.path.join(save_directory, f"{title}_histogram.png")
            save_histogram_image(processed_image, histogram_save_path, title=f"{title} - Processed Image Histogram")
            print(f"{title}: Histogram image saved at {histogram_save_path}.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'i' to save the histogram image of the processed image. The histogram image will be saved in the same directory as the processed images. Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value and the `save_directory` parameter when calling the function according to your preferences.





Certainly! Let's continue by adding the functionality to apply a filter to the processed image. We'll introduce a new key, 'f', to trigger the filter application, and we'll update the `slideshow_compare` function to handle this functionality.

```python
import time
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def apply_filter(image, filter_type='blur', kernel_size=5):
    if filter_type == 'blur':
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image

def display_histogram(image, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def save_histogram_image(image, save_path, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def search_comments(comments, keywords):
    # ... (previous implementation remains the same)

def export_user_inputs(user_inputs, export_filename="user_inputs.csv"):
    # ... (previous implementation remains the same)

def visualize_key_distribution(user_inputs):
    # ... (previous implementation remains the same)

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Record user input
        user_inputs.append(chr(key))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image

_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            distribution_image = visualize_key_distribution(user_inputs)
            cv2.imshow("Key Distribution", distribution_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(key).lower() == 'e':
            # Export user inputs to CSV
            export_filename = input("Enter the export filename (default: user_inputs.csv): ").strip() or "user_inputs.csv"
            export_user_inputs(user_inputs, export_filename)
            print(f"User inputs exported to {export_filename}.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keywords = input("Enter search keywords (comma-separated): ").split(',')
            matching_comments = search_comments(comments, search_keywords)
            if matching_comments:
                print("Matching comments:")
                for match_title, match_comment in matching_comments:
                    print(f"{match_title}: {match_comment}")
            else:
                print("No matching comments found.")
        elif chr(key).lower() == 'h':
            # Display histogram of the processed image
            display_histogram(processed_image, title=f"{title} - Processed Image Histogram")
        elif chr(key).lower() == 'i':
            # Save histogram image
            histogram_save_path = os.path.join(save_directory, f"{title}_histogram.png")
            save_histogram_image(processed_image, histogram_save_path, title=f"{title} - Processed Image Histogram")
            print(f"{title}: Histogram image saved at {histogram_save_path}.")
        elif chr(key).lower() == 'f':
            # Apply filter to the processed image
            filter_type = input("Enter the filter type ('blur', 'gaussian', 'median', 'bilateral'): ").strip().lower()
            kernel_size = int(input("Enter the kernel size (odd number): ").strip())
            processed_images[processed_image_index] = apply_filter(processed_image, filter_type, kernel_size)
            action_stack.append(('f', filter_type, kernel_size))  # Push action to the stack
            print(f"{title}: Filter '{filter_type}' applied with kernel size {kernel_size}.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'f' to apply a filter to the processed image. You'll be prompted to enter the filter type ('blur', 'gaussian', 'median', 'bilateral') and the kernel size (an odd number). Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value and the `save_directory` parameter when calling the function according to your preferences.




Certainly! Let's continue by adding the ability to delete comments for an image. We'll introduce a new key, 'x', to delete a comment, and we'll update the `slideshow_compare` function accordingly.

```python
import time
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def apply_filter(image, filter_type='blur', kernel_size=5):
    if filter_type == 'blur':
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image

def display_histogram(image, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def save_histogram_image(image, save_path, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def search_comments(comments, keywords):
    # ... (previous implementation remains the same)

def export_user_inputs(user_inputs, export_filename="user_inputs.csv"):
    # ... (previous implementation remains the same)

def visualize_key_distribution(user_inputs):
    # ... (previous implementation remains the same)

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Record user input
        user_inputs.append(chr(key))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images

[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                elif last_action[0] == 'f':
                    # Undo filter application
                    _, last_filter_type, last_kernel_size = last_action
                    processed_images[processed_image_index] = apply_filter(processed_image, last_filter_type, last_kernel_size)
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            distribution_image = visualize_key_distribution(user_inputs)
            cv2.imshow("Key Distribution", distribution_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(key).lower() == 'e':
            # Export user inputs to CSV
            export_filename = input("Enter the export filename (default: user_inputs.csv): ").strip() or "user_inputs.csv"
            export_user_inputs(user_inputs, export_filename)
            print(f"User inputs exported to {export_filename}.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keywords = input("Enter search keywords (comma-separated): ").split(',')
            matching_comments = search_comments(comments, search_keywords)
            if matching_comments:
                print("Matching comments:")
                for match_title, match_comment in matching_comments:
                    print(f"{match_title}: {match_comment}")
            else:
                print("No matching comments found.")
        elif chr(key).lower() == 'h':
            # Display histogram of the processed image
            display_histogram(processed_image, title=f"{title} - Processed Image Histogram")
        elif chr(key).lower() == 'i':
            # Save histogram image
            histogram_save_path = os.path.join(save_directory, f"{title}_histogram.png")
            save_histogram_image(processed_image, histogram_save_path, title=f"{title} - Processed Image Histogram")
            print(f"{title}: Histogram image saved at {histogram_save_path}.")
        elif chr(key).lower() == 'f':
            # Apply filter to the processed image
            filter_type = input("Enter the filter type ('blur', 'gaussian', 'median', 'bilateral'): ").strip().lower()
            kernel_size = int(input("Enter the kernel size (odd number): ").strip())
            processed_images[processed_image_index] = apply_filter(processed_image, filter_type, kernel_size)
            action_stack.append(('f', filter_type, kernel_size))  # Push action to the stack
            print(f"{title}: Filter '{filter_type}' applied with kernel size {kernel_size}.")
        elif chr(key).lower() == 'c':
            # Add or edit a comment
            new_comment = input("Enter a comment for the image: ").strip()
            comments[title] = new_comment
            print(f"{title}: Comment added/edited.")
        elif chr(key).lower() == 'd':
            # Delete a comment
            if title in comments:
                del comments[title]
                print(f"{title}: Comment deleted.")
            else:
                print(f"{title}: No comment to delete.")
        # ... (other key handling remains the same)

        time.sleep(current_interval)  # Pause for the specified interval before displaying the next image

    return user_inputs
```

Now, during the slideshow, you can press 'd' to delete the comment for the current image. The comment will be removed from the dictionary. Feel free to modify the keys and add more functionalities as needed. Adjust the `default_interval` value and the `save_directory` parameter when calling the function according to your preferences.





Certainly! Let's continue by adding the functionality to save and load the current state of the slideshow. We'll introduce new keys, 's' for saving and 'l' for loading, and update the `slideshow_compare` function accordingly.

```python
import time
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import pickle  # Import the pickle module for saving and loading states

def apply_filter(image, filter_type='blur', kernel_size=5):
    if filter_type == 'blur':
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image

def display_histogram(image, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def save_histogram_image(image, save_path, title="Histogram"):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def search_comments(comments, keywords):
    # ... (previous implementation remains the same)

def export_user_inputs(user_inputs, export_filename="user_inputs.csv"):
    # ... (previous implementation remains the same)

def visualize_key_distribution(user_inputs):
    # ... (previous implementation remains the same)

def save_state(state, filename="slideshow_state.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(state, file)

def load_state(filename="slideshow_state.pkl"):
    try:
        with open(filename, 'rb') as file:
            state = pickle.load(file)
        return state
    except FileNotFoundError:
        print(f"Error: State file '{filename}' not found.")
        return None

def slideshow_compare(original_images, processed_images, window_titles=None, comments=None, default_interval=3, save_directory="processed_images"):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(original_images))]

    user_inputs = []
    processed_image_index = 0
    current_interval = default_interval
    action_stack = []  # Stack to keep track of image processing actions

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    while True:
        original_image = original_images[processed_image_index]
        processed_image = processed_images[processed_image_index]
        title = window_titles[processed_image_index]

        key = compare_images(original_image, processed_image, window_title=f"{title} (Original vs Processed)", comment=comments.get(title, None))

        # Record user input
        user_inputs.append(chr(key))

        # Print user input and wait for specific keys for navigation
        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, 'v' to visualize key distribution, 'e' to export comments to CSV, 'k' to search for comments based on keywords, 'h' to display histogram, 'i' to save histogram image, 'f' to apply filter, 'r' to rotate image, 'z' to resize image, 'x' to flip horizontally, 'y' to flip vertically, 'b' to adjust brightness and contrast, 'd' to detect edges, 'u' to undo last action, 't' to save processed images, '+' to increase interval, '-' to decrease interval, or any other key to continue.")

        if chr(key).lower() == 'n':
            # Move to the next image
            processed_image_index = (processed_image_index + 1) % len(original_images)
        elif chr(key).lower() == 'p':
            # Move to the previous image
            processed_image_index = (processed_image_index - 1) % len(original_images)
        elif chr(key).lower() == 'q':
            # Quit the slideshow
            break
        elif chr(key) == '+':
            # Increase the interval
            current_interval += 1
            print(f"Interval increased to {current_interval} seconds.")
        elif chr(key) == '-':
            # Decrease the interval, but ensure it doesn't go below 1 second
            current_interval = max(1, current_interval - 1)
            print(f"Interval decreased to {current_interval} seconds.")
        elif chr(key).lower() == 'r':
            # Rotate the processed image by 90 degrees clockwise
            processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            action_stack.append('r')  # Push action to the stack
            print(f"{title}: Image rotated by 90 degrees clockwise

.")
        elif chr(key).lower() == 'z':
            # Resize the processed image
            resize_factor = float(input("Enter the resize factor (e.g., 0.5 for half size, 2 for double size): "))
            processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=resize_factor, fy=resize_factor)
            action_stack.append(('z', resize_factor))  # Push action to the stack
            print(f"{title}: Image resized by factor {resize_factor}.")
        elif chr(key).lower() == 'x':
            # Flip the processed image horizontally
            processed_images[processed_image_index] = cv2.flip(processed_image, 1)
            action_stack.append('x')  # Push action to the stack
            print(f"{title}: Image flipped horizontally.")
        elif chr(key).lower() == 'y':
            # Flip the processed image vertically
            processed_images[processed_image_index] = cv2.flip(processed_image, 0)
            action_stack.append('y')  # Push action to the stack
            print(f"{title}: Image flipped vertically.")
        elif chr(key).lower() == 'b':
            # Adjust brightness and contrast
            brightness = float(input("Enter the brightness adjustment factor (e.g., 1.0 for no change): "))
            contrast = float(input("Enter the contrast adjustment factor (e.g., 1.0 for no change): "))
            processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=brightness, beta=contrast)
            action_stack.append(('b', brightness, contrast))  # Push action to the stack
            print(f"{title}: Image brightness and contrast adjusted.")
        elif chr(key).lower() == 'd':
            # Detect edges in the processed image
            processed_images[processed_image_index] = cv2.Canny(processed_image, 100, 200)
            action_stack.append('d')  # Push action to the stack
            print(f"{title}: Edges detected in the image.")
        elif chr(key).lower() == 'u':
            # Undo the last action
            if action_stack:
                last_action = action_stack.pop()
                if last_action == 'r':
                    # Undo rotation
                    processed_images[processed_image_index] = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif last_action == 'z':
                    # Undo resizing
                    _, last_resize_factor = action_stack.pop()
                    processed_images[processed_image_index] = cv2.resize(processed_image, None, fx=1/last_resize_factor, fy=1/last_resize_factor)
                elif last_action == 'x':
                    # Undo horizontal flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 1)
                elif last_action == 'y':
                    # Undo vertical flip
                    processed_images[processed_image_index] = cv2.flip(processed_image, 0)
                elif last_action == 'b':
                    # Undo brightness and contrast adjustment
                    _, last_brightness, last_contrast = action_stack.pop()
                    processed_images[processed_image_index] = cv2.convertScaleAbs(processed_image, alpha=1/last_brightness, beta=1/last_contrast)
                elif last_action == 'd':
                    # Undo edge detection
                    processed_images[processed_image_index] = original_images[processed_image_index].copy()
                elif last_action[0] == 'f':
                    # Undo filter application
                    _, last_filter_type, last_kernel_size = last_action
                    processed_images[processed_image_index] = apply_filter(processed_image, last_filter_type, last_kernel_size)
                print(f"{title}: Last action undone.")
        elif chr(key).lower() == 't':
            # Save the processed images
            save_path = os.path.join(save_directory, f"{title}_processed.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"{title}: Processed image saved at {save_path}.")
        elif chr(key).lower() == 'v':
            # Visualize key distribution
            distribution_image = visualize_key_distribution(user_inputs)
            cv2.imshow("Key Distribution", distribution_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(key).lower() == 'e':
            # Export user inputs to CSV
            export_filename = input("Enter the export filename (default: user_inputs.csv): ").strip() or "user_inputs.csv"
            export_user_inputs(user_inputs, export_filename)
            print(f"User inputs exported to {export_filename}.")
        elif chr(key).lower() == 'k':
            # Search for comments based on keywords
            search_keywords = input("Enter search keywords (comma-separated): ").split(',')
            matching_comments = search_comments(comments, search_keywords)
            if matching_comments:
                print("Matching comments:")
                for match_title, match_comment in matching_comments:
                    print(f"{match_title}: {match_comment}")
            else:
                print("No matching comments found.")
        elif chr(key).lower() == 'h':
            # Display histogram of the processed image
            display_histogram(processed_image, title=f"{title} - Processed Image Histogram")
        elif chr(key).lower() == 'i':
            # Save histogram
```
