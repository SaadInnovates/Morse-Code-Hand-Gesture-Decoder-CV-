import os

# Replace this with your target folder
folder_path = "Data/start"

# List of image extensions to consider
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Count images
image_count = 0
for file in os.listdir(folder_path):
    if file.lower().endswith(image_extensions):
        image_count += 1

print(f"Total images in '{folder_path}': {image_count}")