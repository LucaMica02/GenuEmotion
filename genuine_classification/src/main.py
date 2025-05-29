import cv2
import pandas as pd
import subprocess
import os
from moviepy import ImageClip
from predictor import predict_au, predict_landmark
from dataset_src.create_landmarks_dataset import get_landmarks

# Delete the files created to extract the Action Units
def clean_up(src_path):
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
    # Construct the command
    command = [
        exe_path,
        "-f", src_path,
        "-aus",
        "-out_image",
        "-verbose",
        "-out_dir", output_dir
    ]

    # Run the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Success:")
        print(result.stdout)
        file_path = os.path.join(output_dir, "face.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            row = df.iloc[0].tolist()
            return row[5:22] # Take only the Action Units intensity
        else:
            return []
    except subprocess.CalledProcessError as e:
        print("Error occurred:")
        print(e.stderr)

# Convert a png to mp4 (needed to apply OpenFace -aus)
def png_to_mp4(img_path):
    out_path = img_path.split('.')[0] + ".mp4"
    clip = ImageClip(img_path, duration=1)  # 1 seconds video
    clip.write_videofile(out_path, fps=24)
    print(f"Converted {img_path} to mp4")
    return out_path

# Return the landmarks prediction in text format
def get_landmarks_text(img_path):
    text_land = "Landmarks: "
    landmarks = get_landmarks(img_path)
    if len(landmarks) == 936:
        prediction_land, confidence_land = predict_landmark(landmarks)
        text_land += f"{prediction_land} - {confidence_land:.2f}%"
    else:
        text_land += "No landmarks extracted from the frame"
    return text_land

# Return the aus prediction in text format
def get_aus_text(img_path, exe_path, output_dir):
    text_au = "AUs: "
    out_path = png_to_mp4(img_path)
    row = run_OpenFace(out_path, exe_path, output_dir)
    if row:
        if row == [0.0] * 17:
            text_au += "No AUs detected"
        else:
            prediction_au, confidence_au = predict_au(row)
            text_au += f"{prediction_au} - {confidence_au:.2f}%"
    else:
        text_au += "No data extracted from the frame."
    return text_au

def main():
    # PATHS
    img_path = r"C:\Users\lucam\Drive\Desktop\GenuEmotion\face.png"
    exe_path = r"C:\Users\lucam\Drive\Desktop\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
    output_dir = r"C:\Users\lucam\Drive\Desktop\GenuEmotion"
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    # Use face cascade to draw a rectangle around the face
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: loading face cascade from {FACE_CASCADE_PATH}.")
        return

    # Open default webcam
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Log instructions
    print("Starting webcam capture. \
        \n - Press 'q' to quit. \
        \n - Press 'a' to get genuinity prediction based on Face Action Units \
        \n - Press 'l' to get genuinity prediction based on Face Landmarks"
    )

    while True:
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Draw the rectangle
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        cv2.imshow("Webcam Feed", frame)

        # Take the key pressed from the user
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a') or key == ord('l'): 
            # Save the frame
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")

            # Get the prediction in text format
            if key == ord('a'):
                text = get_aus_text(img_path, exe_path, output_dir)
            else:
                text = get_landmarks_text(img_path)
            
            # Show the frozen frame with prediction
            display_frame = frame.copy()
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Result", display_frame)
            print("Showing result... Press any key to continue.")
            cv2.waitKey(0) 
            cv2.destroyWindow("Result")  # Close only the result window

    cap.release()
    cv2.destroyAllWindows()
    clean_up(img_path)

if __name__ == "__main__":
    main()