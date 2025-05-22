import os
import csv
import pandas as pd
import random

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

def create(root_path, last_path, class_name):
    rows = [[os.path.join(last_path, file), class_name] for file in os.listdir(os.path.join(root_path, last_path))]
    random.shuffle(rows)
    append_rows(rows[:1000], dataset_path)

def shuffle_dataset(src, dst):
    # Load, shuffle and save it
    df = pd.read_csv(src, delimiter=';')
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_shuffled.to_csv(dst, index=False, sep=';')

dataset_path = "C:\\Users\\lucam\\Drive\\Desktop\\GenuEmotion\\genuine_classification\\dataset\\images_dataset.csv"
dataset_path_shuffled = "C:\\Users\\lucam\\Drive\\Desktop\\GenuEmotion\\genuine_classification\\dataset\\images_dataset_shuffled.csv"
root_path = "C:\\Users\\lucam\\Drive\\Desktop\\dataset"

HEADER = ["path", "class"]
create_file(dataset_path, HEADER)
create(root_path, "genuine", 1)
create(root_path, "posed", 0)
shuffle_dataset(dataset_path, dataset_path_shuffled)