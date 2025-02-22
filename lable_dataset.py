import os
import pandas as pd

# Path to your captcha images
captcha_folder = "work"
output_csv = "captcha_labels.csv"

# Collect image paths and labels
data = []
for filename in os.listdir(captcha_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  
        label = os.path.splitext(filename)[0] 
        image_path = os.path.join(captcha_folder, filename)
        data.append([image_path, label])

# Save to CSV
df = pd.DataFrame(data, columns=["image_path", "text"])
df.to_csv(output_csv, index=False)

print(f"Dataset saved to {output_csv}")
