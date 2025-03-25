#NamTranImprov
import os
from os.path import join

from util.img_util import readImageFile, saveImageFile
from util.inpaint_util import removeHair

# file paths
data_dir = './data'
save_dir = './result'

# get the list of image files
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_filenames = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(valid_extensions)])

if not image_filenames:
    print("No image files found in the 'data' directory.")
    exit()

print(f"Found {len(image_filenames)} images.")

for file_name in image_filenames:
    file_path = join(data_dir, file_name)

    # read the image
    img_rgb, img_gray = readImageFile(file_path)

    # hair removal
    blackhat, thresh, img_out = removeHair(img_rgb, img_gray, kernel_size=25, threshold=10)

    # save the output image
    save_file_path = join(save_dir, f'processed_{file_name}')
    saveImageFile(img_out, save_file_path)

    print(f"Processed and saved: {save_file_path}")

print("Processing complete. All images have been saved in the.")
