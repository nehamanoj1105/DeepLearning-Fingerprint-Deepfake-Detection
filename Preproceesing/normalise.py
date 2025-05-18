import os
import cv2
import numpy as np

def normalize_image(image):
    normalized_image = image / 255.0
    return normalized_image

def normalize_images_in_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if not filename.lower().endswith('.bmp'):
            continue
        try:
            img = cv2.imread(file_path) 
            if img is None:
                print(f"Failed to read {filename}, skipping...")
                continue
            normalized_img = normalize_image(img)
            print(f"Min and Max values for {filename} before saving: "
                  f"Min = {normalized_img.min()}, Max = {normalized_img.max()}")
            normalized_img_uint8 = (normalized_img * 255).astype(np.uint8)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, normalized_img_uint8)
            print(f"Normalized and saved {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    input_folder = "resized(224*224)" 
    output_folder = "normalised"

    normalize_images_in_folder(input_folder, output_folder)
