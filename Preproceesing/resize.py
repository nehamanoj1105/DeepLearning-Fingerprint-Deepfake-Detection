import cv2
import os

input_folder = 'Real'
output_folder = 'altered'
target_size = (224, 224) 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.bmp')or filename.endswith('.BMP') or filename.endswith('.tif'):
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized_img = cv2.resize(img, target_size)
            cv2.imwrite(os.path.join(output_folder, filename), resized_img)

print("Resizing complete.")
