import os 
import csv
import pandas as pd

def create_file(filename, header):
    # If not exists create the file and write the headers
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(header)

def get_file_rows(file_path, class_name):
    rows = []
    with open(file_path, "r") as f:
        f.readline() # Skip the first line
        lines = f.readlines()
        for line in lines:
            row = []
            line = line.split(',')
            for i in range(676, 693): # 676 to 692
                #print(i, line[i])
                row.append(float(line[i]))
            row.append(class_name) # 0 for posed, 1 for genuine
            rows.append(row)
    return rows

def get_total_rows(path, class_name):
    rows = []
    files = os.listdir(path)
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(path, file)
            file_rows = get_file_rows(file_path, class_name)
            for row in file_rows:
                rows.append(row)
    return rows

def append_rows(rows, filename):
    # Append the data
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for row in rows:
            writer.writerow(row)

def create_dataset(filename, path, header, class_name):
    create_file(filename, header)
    rows = get_total_rows(path, class_name)
    append_rows(rows, filename)

def shuffle_dataset(src, dst):
    # Load, shuffle and save it
    df = pd.read_csv("src", delimiter=';')
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_shuffled.to_csv(dst, index=False, sep=';')

HEADER = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", 
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", 
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r", "class"
]
path_posed = "C:/Users/lucam/Drive/Desktop/processed_p"
path_genuine = "C:/Users/lucam/Drive/Desktop/processed_g"
filename = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/dataset/dataset.csv"
shuffled_filename = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/dataset/shuffle_dataset.csv"
create_dataset(filename, path_posed, HEADER, 0)
create_dataset(filename, path_genuine, HEADER, 1)
shuffle_dataset(filename, shuffled_filename)