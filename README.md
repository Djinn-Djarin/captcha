# CAPTCHA Generation and Recognition with CNN

This repository demonstrates the process of generating CAPTCHA images and training a Convolutional Neural Network (CNN) to recognize the text within them. The project includes scripts for data labeling, model training, and testing, providing a comprehensive pipeline for CAPTCHA recognition tasks.

## 👂 Project Structure

```
captcha/
├── labeled_images/           # Directory containing labeled CAPTCHA images
├── captcha_labels.csv        # CSV file mapping images to their labels
├── captcha_model.h5          # Trained CNN model
├── label_dataset.py          # Script to generate labeled dataset
├── test_img.py               # Script to test the model on new images
├── train_cnn.py              # Script to train the CNN model
├── pyproject.toml            # Project metadata and dependencies
└── .gitattributes            # Git attributes for handling binary files
```

## ⚙️ Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Djinn-Djarin/captcha.git
cd captcha
pip install -r requirements.txt
```

## 🧢 Usage

### 1. Labeling CAPTCHA Images

To generate a labeled dataset of CAPTCHA images, run:

```bash
python label_dataset.py
```

This script will process the images in the `labeled_images/` directory and create a CSV file (`captcha_labels.csv`) mapping each image to its corresponding label.

### 2. Training the CNN Model

To train the CNN model on the labeled dataset, execute:

```bash
python train_cnn.py
```

This will initiate the training process and save the trained model as `captcha_model.h5`.

### 3. Testing the Model

To test the trained model on a new CAPTCHA image, use:

```bash
python test_img.py --image path_to_image.png
```

Replace `path_to_image.png` with the path to the image you wish to test.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
