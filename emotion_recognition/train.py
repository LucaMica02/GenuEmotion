# train.py
# Questo modulo contiene la funzione principale per l'addestramento del modello.
# Gestisce il ciclo di addestramento, la validazione, il calcolo delle metriche
# e il salvataggio del modello migliore.

import torch
import torch.nn as nn
from tqdm import tqdm # Per visualizzare una barra di progresso durante l'addestramento
import os
from utils import save_checkpoint # Import della funzione per salvare i checkpoint

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir):
    """
    Funzione principale per l'addestramento di un modello PyTorch.

    Args:
        model (nn.Module): Il modello PyTorch da addestrare.
        train_loader (DataLoader): DataLoader per il set di addestramento.
        val_loader (DataLoader): DataLoader per il set di validazione.
        criterion (nn.Module): Funzione di loss (es. CrossEntropyLoss).
        optimizer (Optimizer): Ottimizzatore (es. Adam).
        num_epochs (int): Numero di epoche per l'addestramento.
        device (torch.device): Dispositivo su cui eseguire l'addestramento ('cuda' o 'cpu').
        checkpoint_dir (str): Directory dove salvare i checkpoint del modello.

    Returns:
        tuple: Il modello addestrato e un dizionario contenente la storia dell'addestramento
        (loss e accuratezza per addestramento e validazione).
    """
    best_val_accuracy = 0.0 # Tiene traccia della migliore accuratezza di validazione raggiunta
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    } # Dizionario per salvare le metriche di addestramento

    # Ciclo di addestramento per il numero specificato di epoche
    for epoch in range(num_epochs):
        # -------------------- Fase di Addestramento --------------------
        model.train() # Imposta il modello in modalità addestramento (abilita dropout, batch norm, ecc.)
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # tqdm avvolge l'iteratore per mostrare una barra di progresso
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoca {epoch+1}/{num_epochs} (Addestramento)")):
            inputs, labels = inputs.to(device), labels.to(device) # Sposta input e etichette sul dispositivo

            # Reset dei gradienti dell'ottimizzatore
            optimizer.zero_grad()

            # Forward pass: calcola l'output del modello
            outputs = model(inputs)
            # Calcola la loss
            loss = criterion(outputs, labels)

            # Backward pass: calcola i gradienti
            loss.backward()
            # Aggiorna i pesi del modello
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Accumula la loss del batch
            
            # Calcola l'accuratezza del batch
            _, predicted = torch.max(outputs.data, 1) # Ottiene la classe con la probabilità più alta
            total_train += labels.size(0) # Aggiunge il numero di campioni nel batch al totale
            correct_train += (predicted == labels).sum().item() # Conta le previsioni corrette

        epoch_train_loss = running_loss / len(train_loader.dataset) # Loss media per l'epoca
        epoch_train_accuracy = 100 * correct_train / total_train # Accuratezza per l'epoca

        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_accuracy)

        # -------------------- Fase di Validazione --------------------
        model.eval() # Imposta il modello in modalità valutazione (disabilita dropout, batch norm, ecc.)
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Disabilita il calcolo dei gradienti per la fase di validazione
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoca {epoch+1}/{num_epochs} (Validazione)"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100 * correct_val / total_val

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        # Stampa le metriche dell'epoca
        print(f'Epoca [{epoch+1}/{num_epochs}], '
              f'Loss Addestramento: {epoch_train_loss:.4f}, Acc Addestramento: {epoch_train_accuracy:.2f}%, '
              f'Loss Validazione: {epoch_val_loss:.4f}, Acc Validazione: {epoch_val_accuracy:.2f}%')

        # Salva il modello migliore basato sull'accuratezza di validazione
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            # Salva lo stato del modello, dell'ottimizzatore, l'epoca e la migliore accuratezza
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
            }, filename=os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Checkpoint salvato: Migliore accuratezza di validazione: {best_val_accuracy:.2f}%")

    return model, history # Restituisce il modello addestrato e la storia delle metriche
