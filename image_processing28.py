import cv2
import matplotlib.pyplot as plt
import csv
import json

def display_images(images, window_titles=None, window_size=(800, 600), comments=None):
    if window_titles is None:
        window_titles = [f"Image {i+1}" for i in range(len(images))]

    user_inputs = []

    i = 0  

    while i < len(images):
        image = images[i]
        title = window_titles[i]

        key = display_image(image, window_title=title, window_size=window_size, comment=comments.get(title, None))

        print(f"{title}: Key pressed - {chr(key)}")
        print("Press 'n' for the next image, 'p' for the previous image, 'q' to quit, 'c' to add/edit a comment, 'd' to delete a comment, 's' to save the current state, 'l' to load a state, or any other key to continue.")

        if chr(key).lower() == 'n':
            if i < len(images) - 1:
                i += 1
        elif chr(key).lower() == 'p':
            if i > 0:
                i -= 1
        elif chr(key).lower() == 'q':
            break
        elif chr(key).lower() == 'c':
            comment = input(f"Enter/Edit a comment for {title}: ")
            comments[title] = comment
        elif chr(key).lower() == 'd':
            if title in comments:
                del comments[title]
                print(f"Comment for {title} deleted.")
            else:
                print(f"No comment found for {title}.")
        elif chr(key).lower() == 's':
            save_state(user_inputs, comments, 'saved_state.json')
            print("Current state saved.")
        elif chr(key).lower() == 'l':
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
    resized_image = cv2.resize(image, window_size)

    plt.imshow(resized_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(window_title)

    if comment:
        plt.text(10, 30, f"Comment: {comment}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.show(block=False)

    key = cv2.waitKey(0) & 0xFF

    plt.close()

    return key

# ... (rest of the code remains the same)

image_paths = ['images/image.jpg', 'images/image.jpeg']
csv_file = 'user_inputs.csv'
summary_file = 'user_inputs_summary.txt'

loaded_user_inputs, loaded_comments = load_from_csv(csv_file)

loaded_summary_user_inputs, loaded_summary_comments = load_summary_from_file(summary_file)

images = [cv2.imread(image_path) for image_path in image_paths]

if all(image is not None for image in images):
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    user_inputs_original, comments_original = display_images(images, window_titles=['Original 1', 'Original 2', 'Original 3'], comments=loaded_comments)
    user_inputs_grayscale, comments_grayscale = display_images(grayscale_images, window_titles=['Grayscale 1', 'Grayscale 2', 'Grayscale 3'], comments=loaded_comments)

    user_inputs = loaded_user_inputs + user_inputs_original + user_inputs_grayscale
    comments = {**loaded_comments, **comments_original, **comments_grayscale}

    save_to_csv(user_inputs, comments, csv_file)

    prompt_user()

    for title

, key in user_inputs:
        print(f"Key pressed for {title}: {chr(key)}")

    for title, comment in comments.items():
        print(f"Comment for {title}: {comment}")

    display_summary(user_inputs, comments)

    save_summary_to_file(user_inputs, comments)

    display_summary(loaded_summary_user_inputs, loaded_summary_comments)

    update_summary_comments(user_inputs, comments)

    display_summary(user_inputs, comments)

    delete_entry(user_inputs, comments)

    display_summary(user_inputs, comments)

    delete_entry_from_summary(user_inputs, comments)

    display_summary(user_inputs, comments)

    navigate_summary(user_inputs, comments, images)

    clear_summary_comments(comments)

    display_summary(user_inputs, comments)

    save_state(user_inputs, comments, 'final_state.json')
else:
    print("Error: Unable to load one or more images.")