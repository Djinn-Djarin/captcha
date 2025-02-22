import base64
import os

def base64_to_jpg(base64_string, count):
    # Define the folder path (relative to the script's directory)
    output_folder = "img_folder"  # No leading `/`, keeps it in the current directory

    # Create the folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the output filename
    output_filename = os.path.join(output_folder, f"{count}_output.jpg")

    # Extract the actual Base64 data (remove prefix like "data:image/jpg;base64,")
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode and save the image
    try:
        image_data = base64.b64decode(base64_string)
        with open(output_filename, "wb") as image_file:
            image_file.write(image_data)
        print(f"✅ Image saved as: {output_filename}")
    except Exception as e:
        print(f"❌ Error saving image: {e}")


# ollama run llava "What is the text in this CAPTCHA?" < "/home/server/Documents/git_hala/img_folder/6_output.jpg"
