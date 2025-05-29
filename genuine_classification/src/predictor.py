import torch
from .model import MLPClassifier

# Function to perform prediction using a trained model
def predict(model, best_model_path, input):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model weights from the saved file
    model = MLPClassifier(input_dim=len(input))
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device) 
        logits = model(x) # Forward pass through the model
        probs = torch.sigmoid(logits) # Apply sigmoid to convert logits to probability
        pred = (probs > 0.5).item() # Make a binary prediction (threshold at 0.5)

    return pred, probs.item()

# Function to perform prediction based on the AUs
def predict_au(input):
    best_model_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/au_classifier_best.pth"
    model = MLPClassifier(input_dim=len(input))
    return predict(model, best_model_path, input)

# Function to perform prediction based on the landmarks
def predict_landmark(input):
    best_model_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/land_classifier_best.pth"
    model = MLPClassifier(input_dim=len(input))
    return predict(model, best_model_path, input)