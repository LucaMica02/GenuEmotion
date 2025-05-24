# data_loader.py
# Questo modulo si occupa del caricamento e della pre-elaborazione del dataset FER-2013.

import pandas as pd # Per la manipolazione dei dati CSV
import numpy as np # Per operazioni numeriche, in particolare per le immagini
from torch.utils.data import Dataset # Classe base per i dataset PyTorch
from PIL import Image # Per la manipolazione delle immagini

class FER2013Dataset(Dataset):
    """
    Dataset personalizzato per il dataset FER-2013.
    Carica i dati da un file CSV, estrae le immagini e le etichette,
    e applica le trasformazioni specificate.
    """
    def __init__(self, csv_file, phase, transform=None):
        """
        Costruttore della classe FER2013Dataset.

        Args:
            csv_file (str): Percorso del file CSV del dataset FER-2013.
            phase (str): Fase del dataset da caricare ('Training', 'PublicTest', 'PrivateTest').
            transform (callable, optional): Trasformazioni da applicare alle immagini.
        """
        self.df = pd.read_csv(csv_file) # Carica il file CSV in un DataFrame pandas
        self.df = self.df[self.df['Usage'] == phase] # Filtra il DataFrame in base alla fase (Training, PublicTest, PrivateTest)
        self.transform = transform # Salva le trasformazioni da applicare

        # Mappa le etichette numeriche delle emozioni ai nomi standard delle emozioni
        self.emotion_map = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }

    def __len__(self):
        """
        Restituisce il numero totale di campioni nel dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Recupera un singolo campione (immagine ed etichetta) dal dataset dato un indice.

        Args:
            idx (int): Indice del campione da recuperare.

        Returns:
            tuple: Un tuple contenente il tensore dell'immagine e l'etichetta dell'emozione.
        """
        # Estrai la riga corrispondente all'indice
        row = self.df.iloc[idx]
        
        # Estrai l'etichetta dell'emozione
        emotion = row['emotion']
        
        # Estrai i pixel dell'immagine come stringa e convertili in un array NumPy
        # I pixel sono una stringa di numeri separati da spazi
        pixels_str = row['pixels'].split(' ')
        # Converte la stringa di pixel in un array NumPy di interi
        # e lo rimodella a 48x48 pixel (dimensione delle immagini in FER-2013)
        image = np.array(pixels_str, dtype='uint8').reshape(48, 48)

        # Applica le trasformazioni se specificate
        if self.transform:
            # Le trasformazioni di torchvision si aspettano un'immagine PIL o un tensore.
            # Qui passiamo l'array NumPy direttamente, torchvision.transforms.ToPILImage() lo gestirà.
            image = self.transform(image)

        return image, emotion # Restituisce l'immagine trasformata e l'etichetta
