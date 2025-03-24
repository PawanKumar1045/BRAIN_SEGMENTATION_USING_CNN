import zipfile
import os
import numpy as np
import h5py
import glob
import scipy.io
import cv2

zip_path = r"D:\Downloads\brainTumorDataPublic_15332298.zip"  # Update this with ypur filepath
extract_folder = r"D:\Downloads\BTD3"
output_folder = r"D:\Downloads\OPBTD3"
os.makedirs(extract_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
images, masks, labels = [], [], []
file_list = glob.glob(os.path.join(extract_folder, "*.mat"))
TARGET_SHAPE = (256, 256)
for file_path in file_list:
    try:
        with h5py.File(file_path, 'r') as mat_data:
            print(f"Processing {file_path} (MAT v7.3)...")
            image = np.array(mat_data["cjdata/image"])
            tumor_mask = np.array(mat_data["cjdata/tumorMask"])
            label = int(np.array(mat_data["cjdata/label"])[0][0])

    except OSError:
        print(f"Processing {file_path} (MAT v7.0)...")
        mat_data = scipy.io.loadmat(file_path)
        image = np.array(mat_data["cjdata"]["image"][0, 0])
        tumor_mask = np.array(mat_data["cjdata"]["tumorMask"][0, 0])
        label = int(mat_data["cjdata"]["label"][0, 0][0, 0])
    image = (image - image.min()) / (image.max() - image.min())
    image = cv2.resize(image, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
    tumor_mask = cv2.resize(tumor_mask, TARGET_SHAPE, interpolation=cv2.INTER_NEAREST)
    images.append(image)
    masks.append(tumor_mask)
    labels.append(label)
images = np.array(images, dtype=np.float32) 
masks = np.array(masks, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)
np.save(os.path.join(output_folder, "images.npy"), images)
np.save(os.path.join(output_folder, "masks.npy"), masks)
np.save(os.path.join(output_folder, "labels.npy"), labels)

print("Files saved in:", output_folder)
