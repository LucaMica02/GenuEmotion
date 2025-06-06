import mediapipe as mp
import cv2
import csv
import os
import pandas as pd

def create_file(filename, header):
    # If not exists create the file and write the headers
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(header)

def append_rows(rows, filename):
    # Append the data
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for row in rows:
            writer.writerow(row)

# Take and image path and return the face landmarks extracted using mediapipe
def get_landmarks(img_path, size=(224, 224)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, size)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

    # Store all the landmarks in the format x, y
    landmarks = []
    results = face_mesh.process(rgb_img)
    if results.multi_face_landmarks:
        for facial_landmarks in results.multi_face_landmarks:
            for pt in facial_landmarks.landmark:
                x = pt.x
                y = pt.y
                landmarks.append(x)
                landmarks.append(y)
    return landmarks

# Get the row for every image and then append it to the dataset (up to 1000 rows)
def create(root_path, class_name, dataset_path):
    rows = []
    for file in os.listdir(root_path):
        path = os.path.join(root_path, file)
        row = get_landmarks(path)
        if row:
            row.append(class_name)
            rows.append(row)
        if len(rows) == 1000:
            break
    append_rows(rows, dataset_path)

def shuffle_dataset(src, dst):
    # Load, shuffle and save it
    df = pd.read_csv(src, delimiter=';')
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_shuffled.to_csv(dst, index=False, sep=';')

def create_dataset():
    posed_path = "C:/Users/lucam/Drive/Desktop/dataset/posed"
    gen_path = "C:/Users/lucam/Drive/Desktop/dataset/genuine"
    dataset_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/dataset/landmarks_dataset.csv"
    dataset_path_shuffled = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/dataset/landmarks_dataset_shuffled.csv"
    HEADER = []
    for i in range(468):
        HEADER.append(f"x_{i}")
        HEADER.append(f"y_{i}")
    HEADER.append("class")
    create_file(dataset_path, HEADER)
    create(posed_path, 0, dataset_path)
    create(gen_path, 1, dataset_path)
    shuffle_dataset(dataset_path, dataset_path_shuffled)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

if __name__ == "__main__":
    create_dataset()