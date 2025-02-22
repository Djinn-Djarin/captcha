import os
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = "/home/server/Documents/git_hala/captcha/captcha_model.h5"
model = tf.keras.models.load_model(model_path)

# Constants
IMG_WIDTH, IMG_HEIGHT = 200, 80
CHAR_SET = "abcdefghijklmnopqrstuvwxyz0123456789"
NUM_CLASSES = len(CHAR_SET)
NUM_CHARACTERS = 6  # Same as training

# Character mapping
char_to_index = {char: idx for idx, char in enumerate(CHAR_SET)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Function to preprocess the image
def preprocess_image(img_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    if not os.path.exists(img_path):
        print(f"Warning: File not found '{img_path}'")
        return None  # Skip missing files

    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Unable to read '{img_path}'")
        return None

    h, w = image.shape[:2]

    # Remove diagonal line (bottom-left to top-right)
    cv2.line(image, (0, h), (w, 0), (255, 255, 255), thickness=2)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (inverted for better text contrast)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological dilation to enhance letters
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Resize image to the expected size
    dilated = cv2.resize(dilated, (img_width, img_height))

    # Normalize pixel values
    processed_image = dilated / 255.0

    # Add a channel dimension (needed for CNN input)
    processed_image = np.expand_dims(processed_image, axis=-1)  # Shape: (80, 200, 1)

    return processed_image


# Function to decode model output
def decode_prediction(pred):
    pred_indices = np.argmax(pred, axis=-1)  # Get index with highest probability
    decoded_text = "".join(index_to_char[idx] for idx in pred_indices)
    return decoded_text

# Test the model with an image
test_image_path = "/home/server/Documents/git_hala/git_hala/test_images/Untitled9.jpeg"  # Change this to your test image

image = preprocess_image(test_image_path)

if image is not None:
    image = np.expand_dims(image, axis=0)  # Add batch dimension -> Shape: (1, 80, 200, 1)
    prediction = model.predict(image)
    captcha_text = decode_prediction(prediction[0])  # Decode first sample
    print(f"Predicted CAPTCHA text: {captcha_text}")

