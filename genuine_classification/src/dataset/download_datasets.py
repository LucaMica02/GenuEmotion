import kagglehub
import os
import matplotlib.pyplot as plt
import cv2
import shutil

def download_dataset_posed():
  save_dir = '/content/posed'
  # Create directory if it doesn't exist
  os.makedirs(save_dir, exist_ok=True)

  # Download latest version
  path = kagglehub.dataset_download("sudarshanvaidya/random-images-for-face-emotion-recognition")
  directories = os.listdir(path)

  TOTAL_SIZE = 2000
  SIZE = TOTAL_SIZE // len(directories)

  for dir in directories:
    filespath = os.path.join(path, dir)
    files = os.listdir(filespath)
    print("DIR: ", dir, len(files))
    for i in range(min(SIZE, len(files))):
      img_path = os.path.join(filespath, files[i])
      img = cv2.imread(img_path)
      save_path = os.path.join(save_dir, f"{dir}_{i}.png")
      cv2.imwrite(save_path, img)

  # Zip the directory
  shutil.make_archive('/content/posed', 'zip', '/content/posed')

def download_dataset_genuine():
  save_dir = '/content/genuine'
  # Create directory if it doesn't exist
  os.makedirs(save_dir, exist_ok=True)

  # Download latest version
  path = kagglehub.dataset_download("freak2209/face-data")
  images_path = os.path.join(path, "Custom_Data", "images", "train")
  images = os.listdir(images_path)
  for i in range(len(images)):
      # READ THE IMAGE
      img_path = os.path.join(images_path, images[i])
      img = cv2.imread(img_path, 0)
      save_path = os.path.join(save_dir, f"gen_{i}.png")
      cv2.imwrite(save_path, img)

  # Zip the directory
  shutil.make_archive('/content/genuine', 'zip', '/content/genuine')