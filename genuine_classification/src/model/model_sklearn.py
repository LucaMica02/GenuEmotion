import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load your AU dataset
dataset_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/dataset/AUs_dataset_shuffled.csv"
#dataset_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/dataset/landmarks_dataset_shuffled.csv"
df = pd.read_csv(dataset_path, delimiter=';')

X = df.drop("class", axis=1) 
y = df["class"]

model = RandomForestClassifier()
accuracy = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {accuracy.mean():.3f}")