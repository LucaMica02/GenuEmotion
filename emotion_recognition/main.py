# main.py
# Questo file è il punto di ingresso principale del progetto.
# Si occupa di configurare l'ambiente, caricare i dati, inizializzare il modello,
# avviare il processo di addestramento e valutare le prestazioni finali.

'''COMANDO PER ESEGUIRE IL CODICE DA TERMINALE: python main.py'''

# Import le librerie necessarie
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Import i moduli personalizzati
from data_loader import FER2013Dataset # Modulo per il caricamento dei dati
from model import EmotionCNN # Modulo per la definizione del modello
from train import train_model # Modulo per la funzione di addestramento
from utils import save_checkpoint, load_checkpoint, plot_metrics # Modulo per le utility

# Definizione dei percorsi del dataset e dei checkpoint
DATA_PATH = 'data/fer2013.csv'
CHECKPOINT_DIR = 'checkpoints'
PLOTS_DIR = 'plots'

# Crea le directory se non esistono
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Hyperparameters (Parametri che controllano il processo di addestramento)
NUM_EPOCHS = 50 # Numero di volte che l'intero dataset verrà passato attraverso la rete
BATCH_SIZE = 64 # Numero di campioni elaborati prima di aggiornare i pesi del modello
LEARNING_RATE = 0.001 # Tasso di apprendimento per l'ottimizzatore
NUM_CLASSES = 7 # Numero di classi di emozioni (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

def main():
    # 1. Configurazione del dispositivo (CPU/GPU)
    # Controlla se è disponibile una GPU (CUDA) e la utilizza, altrimenti usa la CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # 2. Definizione delle trasformazioni per le immagini
    # Queste trasformazioni vengono applicate alle immagini prima di essere passate al modello.
    # Sono essenziali per la normalizzazione e l'aumento dei dati.
    train_transforms = transforms.Compose([
        transforms.ToPILImage(), # Converte l'array NumPy in immagine PIL
        transforms.Resize((48, 48)), # Ridimensiona tutte le immagini a 48x48 pixel (dimensione standard per FER-2013)
        transforms.RandomHorizontalFlip(), # Applica un flip orizzontale casuale (aumento dei dati)
        transforms.RandomRotation(10), # Applica una rotazione casuale di +/- 10 gradi (aumento dei dati)
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Modifica casuale di luminosità, contrasto, saturazione, tonalità
        transforms.ToTensor(), # Converte l'immagine PIL in un tensore PyTorch (scala i pixel da [0, 255] a [0, 1])
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizza i valori dei pixel a un intervallo di [-1, 1]
    ])

    val_test_transforms = transforms.Compose([
        transforms.ToPILImage(), # Converte l'array NumPy in immagine PIL
        transforms.Resize((48, 48)), # Ridimensiona a 48x48 pixel
        transforms.ToTensor(), # Converte in tensore
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizza
    ])

    # 3. Caricamento del Dataset
    # Inizializza i dataset per l'addestramento, la validazione e il test.
    # Il dataset FER2013Dataset si occuperà di leggere il file CSV e preparare i dati.
    print("Caricamento del dataset FER-2013...")
    train_dataset = FER2013Dataset(csv_file=DATA_PATH, phase='Training', transform=train_transforms)
    val_dataset = FER2013Dataset(csv_file=DATA_PATH, phase='PublicTest', transform=val_test_transforms)
    test_dataset = FER2013Dataset(csv_file=DATA_PATH, phase='PrivateTest', transform=val_test_transforms)

    # Inizializza i DataLoader
    # I DataLoader permettono di iterare sul dataset in batch e di caricare i dati in parallelo.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Dataset caricato: Training={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    # 4. Inizializzazione del Modello
    # Crea un'istanza della nostra rete neurale EmotionCNN.
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device) # Sposta il modello sul dispositivo (GPU/CPU)
    print("Modello inizializzato:")
    print(model)

    # 5. Definizione della funzione di Loss e dell'Ottimizzatore
    # La funzione di loss misura quanto le previsioni del modello sono lontane dalla verità.
    # L'ottimizzatore aggiorna i pesi del modello per minimizzare la loss.
    criterion = nn.CrossEntropyLoss() # Funzione di loss per la classificazione multi-classe
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Ottimizzatore Adam, popolare per la sua efficacia

    # 6. Addestramento del Modello
    # Chiama la funzione di addestramento definita in train.py.
    print("Inizio addestramento del modello...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        device,
        CHECKPOINT_DIR
    )
    print("Addestramento completato.")

    # 7. Valutazione finale sul Test Set
    # Dopo l'addestramento, valuta le prestazioni del modello sul set di test,
    # che non è stato utilizzato durante l'addestramento o la validazione.
    print("Valutazione finale sul Test Set...")
    model.eval() # Imposta il modello in modalità valutazione (disabilita dropout, batch norm, ecc.)
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): # Disabilita il calcolo dei gradienti per risparmiare memoria e velocizzare
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_dataset)
    test_accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # 8. Plotting delle metriche di addestramento
    # Visualizza l'andamento della loss e dell'accuratezza durante l'addestramento.
    plot_metrics(history, PLOTS_DIR)
    print(f"Grafici di addestramento salvati in {PLOTS_DIR}")

if __name__ == '__main__':
    main()
