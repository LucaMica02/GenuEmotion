# inference.py
# Questo modulo permette di utilizzare il modello addestrato per il riconoscimento delle emozioni
# in tempo reale tramite webcam, utilizzando OpenCV per la cattura video e il rilevamento dei volti.

'''COMANDO PER ESEGUIRE IL CODICE DA TERMINALE: python inference.py'''

import cv2 # Libreria OpenCV per la visione artificiale
import torch
import torch.nn.functional as F
from torchvision import transforms # Per le trasformazioni delle immagini
import numpy as np # Per operazioni numeriche
import pandas as pd
import os
import subprocess
from moviepy import ImageClip
from genuineness_detection.src.predictor import predict_au, predict_landmark
from genuineness_detection.dataset_src.create_landmarks_dataset import get_landmarks

# Importa il modello personalizzato e le utility
from emotion_recognition.model import EmotionCNN
from emotion_recognition.utils import load_checkpoint

# Definizione dei percorsi
CHECKPOINT_PATH = 'checkpoints/best_model.pth' # Percorso del modello addestrato
# Percorso del classificatore Haar Cascade per il rilevamento dei volti
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Mappa le etichette numeriche delle emozioni ai nomi standard delle emozioni
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
    Pre-elabora l'immagine del volto per renderla compatibile con l'input del modello.

    Args:
        face_image (numpy.ndarray): Immagine del volto (array NumPy).

    Returns:
        torch.Tensor: Tensore dell'immagine pre-elaborata.
    """
    # Trasformazioni da applicare all'immagine del volto rilevato
    transform = transforms.Compose([
        transforms.ToPILImage(), # Converte l'array NumPy in immagine PIL
        transforms.Resize((48, 48)), # Ridimensiona a 48x48 pixel
        transforms.ToTensor(), # Converte in tensore (scala a [0, 1])
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizza a [-1, 1]
    ])
    return transform(face_image).unsqueeze(0) # Aggiunge una dimensione per il batch (batch_size=1)

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
    img_path = r"C:\Users\lucam\Drive\Desktop\GenuEmotion\face.png"
    exe_path = r"C:\Users\lucam\Drive\Desktop\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
    output_dir = r"C:\Users\lucam\Drive\Desktop\GenuEmotion"

    # 1. Configurazione del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # 2. Caricamento del modello addestrato
    # Inizializza il modello con il numero corretto di classi
    model = EmotionCNN(num_classes=len(EMOTION_LABELS)).to(device)
    # Carica lo stato del modello dal checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        # load_checkpoint restituisce più valori, ma qui ci interessa solo il modello
        model, _, _, _ = load_checkpoint(model, None, CHECKPOINT_PATH)
        model.eval() # Imposta il modello in modalità valutazione
        print(f"Modello caricato con successo da {CHECKPOINT_PATH}")
    else:
        print(f"Errore: Nessun checkpoint del modello trovato a {CHECKPOINT_PATH}.")
        print("Assicurati di aver addestrato il modello eseguendo main.py prima.")
        return

    # 3. Caricamento del classificatore di volti Haar Cascade
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Errore: Impossibile caricare il classificatore di volti da {FACE_CASCADE_PATH}.")
        print("Assicurati che il file 'haarcascade_frontalface_default.xml' esista.")
        return

    # 4. Inizializzazione della webcam
    cap = cv2.VideoCapture(0) # 0 indica la webcam predefinita
    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam.")
        return

    print("Inizio inferenza in tempo reale. Premi 'q' per uscire.")

    while True:
        ret, frame = cap.read() # Legge un frame dalla webcam
        if not ret:
            print("Errore: Impossibile leggere il frame.")
            break

        # Converte il frame in scala di grigi per il rilevamento dei volti e l'input del modello
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rileva i volti nel frame
        # scaleFactor: quanto l'immagine viene ridotta ad ogni scala di immagine.
        # minNeighbors: quanti vicini ogni rettangolo candidato dovrebbe avere per ritenerlo un volto.
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Disegna un rettangolo attorno al volto rilevato
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Colore blu, spessore 2

            # Estrai la regione del volto (ROI)
            face_roi = gray_frame[y:y+h, x:x+w]

            # Pre-elabora il volto per l'input del modello
            preprocessed_face = preprocess_face(face_roi).to(device)

            # Esegui l'inferenza (previsione dell'emozione)
            with torch.no_grad(): # Disabilita il calcolo dei gradienti
                outputs = model(preprocessed_face)
                # Applica softmax per ottenere le probabilità
                probabilities = F.softmax(outputs, dim=1)
                # Ottieni la classe con la probabilità più alta
                _, predicted_idx = torch.max(probabilities, 1)
                predicted_emotion = EMOTION_LABELS[predicted_idx.item()]
                confidence = probabilities[0, predicted_idx.item()].item() * 100 # Confidenza in percentuale

            # Visualizza l'emozione e la confidenza sul frame
            text = f"{predicted_emotion} ({confidence:.2f}%)"
            # Posiziona il testo sopra il rettangolo del volto
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA) # Colore verde

        # Mostra il frame risultante
        cv2.imshow('Riconoscimento Emozioni', frame)

        # Esci premendo 'q'
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

    # Rilascia la webcam e distruggi tutte le finestre di OpenCV
    cap.release()
    cv2.destroyAllWindows()
    clean_up(img_path)

if __name__ == '__main__':
    main()