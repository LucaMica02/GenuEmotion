# utils.py
# Questo modulo contiene funzioni di utilità per il progetto,
# come il salvataggio e il caricamento dei checkpoint del modello
# e la visualizzazione delle metriche di addestramento.

import torch
import matplotlib.pyplot as plt
import os

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Salva lo stato corrente del modello e dell'ottimizzatore come checkpoint.

    Args:
        state (dict): Un dizionario contenente lo stato da salvare (es. model_state_dict, optimizer_state_dict).
        filename (str): Il percorso e il nome del file dove salvare il checkpoint.
    """
    print(f"Salvataggio checkpoint a: {filename}")
    torch.save(state, filename)

def load_checkpoint(model, optimizer=None, filename="checkpoint.pth.tar"):
    """
    Carica un checkpoint salvato per riprendere l'addestramento o per l'inferenza.

    Args:
        model (nn.Module): Il modello PyTorch a cui caricare lo stato.
        optimizer (Optimizer, optional): L'ottimizzatore a cui caricare lo stato. Default: None.
        filename (str): Il percorso e il nome del file del checkpoint da caricare.

    Returns:
        tuple: Il modello aggiornato, l'ottimizzatore aggiornato (o None), l'epoca di ripartenza
        e la migliore accuratezza di validazione dal checkpoint.
    """
    print(f"Caricamento checkpoint da: {filename}")
    # Mappa i tensori caricati sulla CPU se non è disponibile una GPU
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))

    # Carica lo stato del modello
    model.load_state_dict(checkpoint['model_state_dict'])

    # Carica lo stato dell'ottimizzatore SOLO SE è stato fornito un ottimizzatore
    # e se lo stato dell'ottimizzatore è presente nel checkpoint
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Stato dell'ottimizzatore caricato.")
    elif optimizer is not None and 'optimizer_state_dict' not in checkpoint:
        print("Avviso: L'ottimizzatore è stato fornito ma non è stato trovato nessun 'optimizer_state_dict' nel checkpoint.")


    # Usa .get() per recuperare i valori in modo più sicuro, fornendo un default se non presenti
    epoch = checkpoint.get('epoch', 0) # Default a 0 se 'epoch' non è presente
    best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0) # Default a 0.0 se 'best_val_accuracy' non è presente

    print(f"Checkpoint caricato. Riprendo dall'epoca {epoch}, migliore accuratezza di validazione: {best_val_accuracy:.2f}%")
    return model, optimizer, epoch, best_val_accuracy

def plot_metrics(history, plots_dir):
    """
    Genera e salva i grafici della loss e dell'accuratezza di addestramento e validazione.

    Args:
        history (dict): Dizionario contenente le liste di 'train_loss', 'train_accuracy',
                        'val_loss', 'val_accuracy'.
        plots_dir (str): Directory dove salvare i grafici.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Grafico della Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1) # 1 riga, 2 colonne, 1° subplot
    plt.plot(epochs, history['train_loss'], 'b', label='Loss di Addestramento')
    plt.plot(epochs, history['val_loss'], 'r', label='Loss di Validazione')
    plt.title('Loss di Addestramento e Validazione')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Grafico dell'Accuratezza
    plt.subplot(1, 2, 2) # 1 riga, 2 colonne, 2° subplot
    plt.plot(epochs, history['train_accuracy'], 'b', label='Accuratezza di Addestramento')
    plt.plot(epochs, history['val_accuracy'], 'r', label='Accuratezza di Validazione')
    plt.title('Accuratezza di Addestramento e Validazione')
    plt.xlabel('Epoche')
    plt.ylabel('Accuratezza (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adatta automaticamente i subplot per evitare sovrapposizioni
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png')) # Salva il grafico
    plt.close() # Chiude la figura per liberare memoria
    print(f"Grafici delle metriche di addestramento salvati in {os.path.join(plots_dir, 'training_metrics.png')}")
