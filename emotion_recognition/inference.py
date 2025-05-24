# inference.py
# Questo modulo permette di utilizzare il modello addestrato per il riconoscimento delle emozioni
# in tempo reale tramite webcam, utilizzando OpenCV per la cattura video e il rilevamento dei volti.

'''COMANDO PER ESEGUIRE IL CODICE DA TERMINALE: python inference.py'''

import cv2 # Libreria OpenCV per la visione artificiale
import torch
import torch.nn.functional as F
from torchvision import transforms # Per le trasformazioni delle immagini
import numpy as np # Per operazioni numeriche
import os

# Importa il modello personalizzato e le utility
from model import EmotionCNN
from utils import load_checkpoint

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

def main():
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia la webcam e distruggi tutte le finestre di OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

'''
CODICE DA USARE SE VOGLIO USARE UN'IMMAGINE DA TESTARE ANZICHE' LA WEBCAM (SOSTITUIRE IL CODICE PER INTERO)
COMANDO PER ESEGUIRE DA TERMINALE: python inference.py --image_path "immagini/mia_foto.jpg"

# inference.py
# Questo modulo permette di utilizzare il modello addestrato per il riconoscimento delle emozioni
# su un'immagine statica fornita dall'utente, utilizzando OpenCV per il rilevamento dei volti.

import cv2 # Libreria OpenCV per la visione artificiale
import torch
import torch.nn.functional as F
from torchvision import transforms # Per le trasformazioni delle immagini
import numpy as np # Per operazioni numeriche
import os
import argparse # Per gestire gli argomenti da linea di comando

# Importa il modello personalizzato e le utility
from model import EmotionCNN
from utils import load_checkpoint

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

def main():
    # 1. Parsing degli argomenti da linea di comando
    parser = argparse.ArgumentParser(description="Riconoscimento delle emozioni su un'immagine statica.")
    parser.add_argument('--image_path', type=str, required=True,
                        help="Percorso dell'immagine da analizzare (es. 'immagini/volto.jpg')")
    args = parser.parse_args()

    # 2. Configurazione del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # 3. Caricamento del modello addestrato
    model = EmotionCNN(num_classes=len(EMOTION_LABELS)).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model, _, _, _ = load_checkpoint(model, None, CHECKPOINT_PATH)
        model.eval() # Imposta il modello in modalità valutazione
        print(f"Modello caricato con successo da {CHECKPOINT_PATH}")
    else:
        print(f"Errore: Nessun checkpoint del modello trovato a {CHECKPOINT_PATH}.")
        print("Assicurati di aver addestrato il modello eseguendo main.py prima.")
        return

    # 4. Caricamento del classificatore di volti Haar Cascade
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Errore: Impossibile caricare il classificatore di volti da {FACE_CASCADE_PATH}.")
        print("Assicurati che il file 'haarcascade_frontalface_default.xml' esista.")
        return

    # 5. Caricamento dell'immagine fornita
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Errore: L'immagine non è stata trovata a {image_path}.")
        return

    frame = cv2.imread(image_path) # Carica l'immagine
    if frame is None:
        print(f"Errore: Impossibile caricare l'immagine da {image_path}. Controlla il percorso e il formato.")
        return

    print(f"Analisi dell'immagine: {image_path}")

    # Converte il frame in scala di grigi per il rilevamento dei volti e l'input del modello
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rileva i volti nel frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("Nessun volto rilevato nell'immagine.")
    else:
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

    # Mostra l'immagine risultante
    cv2.imshow('Riconoscimento Emozioni su Immagine', frame)

    # Attendi che l'utente prema un tasto per chiudere la finestra
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


'''