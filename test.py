import json
import numpy as np
import cv2




def display_json_on_image(json_file, output_file="output.jpg", img_size=(300, 400), font_scale=0.5, font_thickness=1):
    """
    Creates a black image and displays the contents of a JSON file (key-value pairs) as text on it.

    :param json_file: Path to the JSON file containing the dictionary.
    :param output_file: Path to save the generated image.
    :param img_size: Tuple (height, width) of the black image.
    :param font_scale: Scale of the text font.
    :param font_thickness: Thickness of the text font.
    :return: None
    """
    # Load the JSON dictionary
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Ensure the data is a dictionary with boolean values
    if not isinstance(data, dict) or not all(isinstance(v, bool) for v in data.values()):
        raise ValueError("JSON file must contain a dictionary with boolean values.")

    # Create a black image
    height, width = img_size
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Define font and starting position
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(100 * font_scale)  # Adjust line height based on font scale
    x, y = 10, 30  # Starting position for text

    # Render each key-value pair as text
    for key, value in data.items():
        text = f"{key}: {value}"
        cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        y += line_height
        if y + line_height > height:  # Stop if text exceeds image height
            print("Warning: Text exceeds image height. Not all items will be displayed.")
            break

    return image



image = display_json_on_image('/Users/silas.mehrens/Desktop/camara-man/settings/settings.json')
print('image made')
cv2.imshow('Test Settings', image)


cv2.waitKey(20000)

cv2.destroyAllWindows()