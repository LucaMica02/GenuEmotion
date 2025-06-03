# main.py
# This module allows the trained model to be used for real-time emotion recognition
# via webcam, using OpenCV for video capture and face detection.

'''COMMAND TO RUN THE CODE FROM TERMINAL: python main.py''' # Corrected file name

import cv2 # OpenCV library for computer vision
import torch
import torch.nn.functional as F
from torchvision import transforms # For image transformations
import numpy as np # For numerical operations
import pandas as pd
import os
import subprocess
from moviepy import ImageClip # Correct import for ImageClip
from genuineness_detection.src.predictor import predict_au, predict_landmark
from genuineness_detection.dataset_src.create_landmarks_dataset import get_landmarks

# Import the custom model and utilities
from emotion_recognition.model import EmotionCNN
from emotion_recognition.utils import load_checkpoint

# Define paths
CHECKPOINT_PATH = 'checkpoints/best_model.pth' # Path to the trained model
# Path to the Haar Cascade classifier for face detection
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Map numerical emotion labels to standard emotion names
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def preprocess_face(face_image):
    """
    Pre-processes the face image to make it compatible with the model's input.

    Args:
        face_image (numpy.ndarray): Face image (NumPy array).

    Returns:
        torch.Tensor: Pre-processed image tensor.
    """
    # Transformations to apply to the detected face image
    transform = transforms.Compose([
        transforms.ToPILImage(), # Converts the NumPy array to a PIL image
        transforms.Resize((48, 48)), # Resizes to 48x48 pixels
        transforms.ToTensor(), # Converts to a tensor (scales to [0, 1])
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizes to [-1, 1]
    ])
    return transform(face_image).unsqueeze(0) # Adds a dimension for the batch (batch_size=1)

# Delete the files created to extract the Action Units
def clean_up(src_path):
    """
    Deletes temporary files created during Action Unit extraction.

    Args:
        src_path (str): Original path of the image used for processing.
    """
    src_path_org = src_path.split('.')[0]
    files_to_delete = [
        src_path,
        src_path_org + ".mp4",
        src_path_org + ".csv",
        src_path_org + "_of_details.txt"
    ]

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")

def run_OpenFace(src_path, exe_path, output_dir):
    """
    Runs OpenFace to extract Action Units from a video.

    Args:
        src_path (str): Path to the input video file.
        exe_path (str): Path to the OpenFace FeatureExtraction.exe executable.
        output_dir (str): Directory to save OpenFace output files.

    Returns:
        list: A list of Action Unit intensities if successful, otherwise an empty list.
    """
    # Construct the command
    command = [
        exe_path,
        "-f", src_path,
        "-aus", # Enable Action Unit extraction
        "-out_image", # Output images with detected facial landmarks
        "-verbose", # Display verbose output
        "-out_dir", output_dir # Specify output directory
    ]

    # Run the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Success:")
        print(result.stdout)
        file_path = os.path.join(output_dir, os.path.basename(src_path).split('.')[0] + ".csv") # Construct the expected CSV filename
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Ensure there's at least one row and enough columns for AUs
            if not df.empty and len(df.columns) > 21:
                row = df.iloc[0].tolist()
                return row[5:22] # Take only the Action Units intensity (columns 5 to 21)
            else:
                print(f"CSV file found but no data or insufficient columns for AUs: {file_path}")
                return []
        else:
            print(f"OpenFace output CSV file not found: {file_path}")
            return []
    except subprocess.CalledProcessError as e:
        print("Error occurred during OpenFace execution:")
        print(e.stderr)
        return []

# Convert a png to mp4 (needed to apply OpenFace -aus)
def png_to_mp4(img_path):
    """
    Converts a PNG image to an MP4 video, which is required for OpenFace processing.

    Args:
        img_path (str): Path to the input PNG image.

    Returns:
        str: Path to the generated MP4 video file.
    """
    out_path = img_path.split('.')[0] + ".mp4"
    clip = ImageClip(img_path, duration=1)  # Create a 1-second video clip from the image
    clip.write_videofile(out_path, fps=24, logger=None) # Write the video file with 24 FPS, suppress verbose output
    print(f"Converted {img_path} to mp4")
    return out_path

# Return the landmarks prediction in text format
def get_landmarks_text(img_path):
    """
    Extracts facial landmarks from an image and returns the prediction in text format.

    Args:
        img_path (str): Path to the input image.

    Returns:
        str: Text describing the landmark prediction and confidence.
    """
    text_land = "Landmarks: "
    landmarks = get_landmarks(img_path) # Get landmarks using a custom function
    if landmarks is not None and len(landmarks) == 936: # Check if landmarks are extracted and have the expected length
        prediction_land, confidence_land = predict_landmark(landmarks) # Predict based on landmarks
        text_land += f"{prediction_land} - {confidence_land:.2f}%"
    else:
        text_land += "No landmarks extracted from the frame"
    return text_land

# Return the aus prediction in text format
def get_aus_text(img_path, exe_path, output_dir):
    """
    Extracts Action Units from an image (by converting it to MP4) and returns the prediction in text format.

    Args:
        img_path (str): Path to the input image.
        exe_path (str): Path to the OpenFace FeatureExtraction.exe executable.
        output_dir (str): Directory to save OpenFace output files.

    Returns:
        str: Text describing the Action Unit prediction and confidence.
    """
    text_au = "AUs: "
    out_path = png_to_mp4(img_path) # Convert the PNG image to MP4 for OpenFace processing
    row = run_OpenFace(out_path, exe_path, output_dir) # Run OpenFace to get Action Units
    if row:
        if row == [0.0] * 17: # Check if all AUs are zero (no AUs detected)
            text_au += "No AUs detected"
        else:
            prediction_au, confidence_au = predict_au(row) # Predict based on Action Units
            text_au += f"{prediction_au} - {confidence_au:.2f}%"
    else:
        text_au += "No data extracted from the frame."
    return text_au

def main():
    """
    Main function to run the real-time emotion recognition and genuineness detection.
    Initializes the model, webcam, and performs inference on detected faces.
    Allows saving frames to analyze Action Units or landmarks.
    """
    # Define paths for temporary files and OpenFace executable
    img_path = r"C:\Users\lucam\Drive\Desktop\GenuEmotion\face.png" # Path to save captured frames
    exe_path = r"C:\Users\lucam\Drive\Desktop\OpenFace_2.2.0_win_x64\FeatureExtraction.exe" # OpenFace executable path
    output_dir = r"C:\Users\lucam\Drive\Desktop\GenuEmotion" # Directory for OpenFace output

    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use CUDA if available, otherwise CPU
    print(f"Using device: {device}")

    # 2. Load the trained model
    # Initialize the model with the correct number of emotion classes
    model = EmotionCNN(num_classes=len(EMOTION_LABELS)).to(device)
    # Load the model state from the checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        # load_checkpoint returns multiple values, but here we only need the model
        model, _, _, _ = load_checkpoint(model, None, CHECKPOINT_PATH)
        model.eval() # Set the model to evaluation mode (disables dropout, batch normalization updates)
        print(f"Model successfully loaded from {CHECKPOINT_PATH}")
    else:
        print(f"Error: No model checkpoint found at {CHECKPOINT_PATH}.")
        print("Please ensure you have trained the model by running main.py first.")
        return

    # 3. Load the Haar Cascade face classifier
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: Unable to load face classifier from {FACE_CASCADE_PATH}.")
        print("Please ensure the file 'haarcascade_frontalface_default.xml' exists.")
        return

    # 4. Initialize the webcam
    cap = cv2.VideoCapture(0) # 0 indicates the default webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    print("Starting real-time inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read() # Read a frame from the webcam
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Convert the frame to grayscale for face detection and model input
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        # scaleFactor: How much the image is reduced at each image scale.
        # minNeighbors: How many neighbors each candidate rectangle should have to retain it as a face.
        # minSize: Minimum possible object size. Objects smaller than this are ignored.
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue color, thickness 2

            # Extract the Region of Interest (ROI) for the face
            face_roi = gray_frame[y:y+h, x:x+w]

            # Pre-process the face for model input
            preprocessed_face = preprocess_face(face_roi).to(device)

            # Perform inference (emotion prediction)
            with torch.no_grad(): # Disable gradient calculation (as we are only inferring)
                outputs = model(preprocessed_face)
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)
                # Get the class with the highest probability
                _, predicted_idx = torch.max(probabilities, 1)
                predicted_emotion = EMOTION_LABELS[predicted_idx.item()] # Get the emotion name
                confidence = probabilities[0, predicted_idx.item()].item() * 100 # Confidence as a percentage

            # Display the emotion and confidence on the frame
            text = f"{predicted_emotion} ({confidence:.2f}%)"
            # Position the text above the face rectangle
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA) # Green color

        # Show the resulting frame
        cv2.imshow('Emotion Recognition', frame)

        # Exit by pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a') or key == ord('l'):
            # Save the current frame
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")

            # Get the prediction in text format based on the pressed key
            if key == ord('a'): # If 'a' is pressed, get Action Unit prediction
                text = get_aus_text(img_path, exe_path, output_dir)
            else: # If 'l' is pressed, get Landmark prediction
                text = get_landmarks_text(img_path)

            # Show the frozen frame with the prediction
            display_frame = frame.copy() # Create a copy to avoid modifying the live frame
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA) # Display prediction text
            cv2.imshow("Result", display_frame) # Show the result in a new window
            print("Showing result... Press any key to continue.")
            cv2.waitKey(0) # Wait indefinitely until any key is pressed
            cv2.destroyWindow("Result")  # Close only the result window

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    clean_up(img_path) # Clean up temporary files

if __name__ == '__main__':
    main() # Run the main function when the script is executed
