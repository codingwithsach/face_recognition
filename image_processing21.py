import cv2
import matplotlib.pyplot as plt
import csv

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  

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