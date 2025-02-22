import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load CSV file
csv_path = "/home/server/Documents/git_hala/captcha_labels.csv"
df = pd.read_csv(csv_path)
df.columns = ["image_path", "text"]  # Ensure correct column names

# Fix image paths if needed
image_dir = "/home/server/Documents/git_hala/work"  # Adjust as per dataset location
df["image_path"] = df["image_path"].apply(lambda x: os.path.abspath(os.path.join("/home/server/Documents/git_hala", x)))


# Constants
IMG_WIDTH, IMG_HEIGHT = 200, 80
NUM_CHARACTERS = 6  # Assuming 6-character CAPTCHAs
CHAR_SET = "abcdefghijklmnopqrstuvwxyz0123456789"  # Possible characters
NUM_CLASSES = len(CHAR_SET)

# Character mapping
char_to_index = {char: idx for idx, char in enumerate(CHAR_SET)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Function to load and preprocess images
import cv2
import numpy as np
import os

def load_image(img_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
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

    # Resize image
    dilated = cv2.resize(dilated, (img_width, img_height))

    # Normalize pixel values
    processed_image = dilated / 255.0
    
    return processed_image


# Load valid images
X = np.array([img for img in (load_image(img) for img in df["image_path"]) if img is not None])

if len(X) == 0:
    raise ValueError("No valid images found. Check file paths.")

X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)  # Add channel dimension

# Convert labels to integer sequences
def encode_label(label):
    return [char_to_index.get(c, 0) for c in label]  # Use 0 for unknown characters

y = [encode_label(text) for text in df["text"]]

# Pad sequences to fixed length
y = pad_sequences(y, maxlen=NUM_CHARACTERS, padding="post", value=0)

# Convert to categorical (One-Hot)
y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_CHARACTERS * NUM_CLASSES, activation='softmax'),
    layers.Reshape((NUM_CHARACTERS, NUM_CLASSES))  # Reshape to match label encoding
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train the model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1)

# Save model
model.save("captcha_model.h5")

# Load later for prediction
model = tf.keras.models.load_model("captcha_model.h5")
