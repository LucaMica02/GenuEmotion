# model.py
# Questo modulo definisce l'architettura della rete neurale convoluzionale (CNN)
# utilizzando PyTorch.

import torch
import torch.nn as nn
import torch.nn.functional as F # Per funzioni come ReLU, MaxPooling

class EmotionCNN(nn.Module):
    """
    Rete Neurale Convoluzionale (CNN) per la classificazione delle emozioni facciali.
    Architettura con blocchi convoluzionali seguiti da pooling e strati completamente connessi.
    """
    def __init__(self, num_classes=7):
        """
        Costruttore della classe EmotionCNN.

        Args:
            num_classes (int): Il numero di classi di emozioni da prevedere (default 7 per FER-2013).
        """
        super(EmotionCNN, self).__init__() # Chiama il costruttore della classe base nn.Module

        # Blocco Convoluzionale 1
        # Input: 1 canale (immagine in scala di grigi), Output: 64 canali
        # Kernel: 3x3, Padding: 1 (per mantenere la dimensione spaziale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) # Normalizzazione del batch per stabilizzare l'addestramento
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Riduce la dimensione spaziale (es. 48x48 -> 24x24)
        self.dropout1 = nn.Dropout(0.25) # Dropout per prevenire l'overfitting

        # Blocco Convoluzionale 2
        # Input: 64 canali, Output: 128 canali
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Riduce la dimensione spaziale (es. 24x24 -> 12x12)
        self.dropout2 = nn.Dropout(0.25)

        # Blocco Convoluzionale 3
        # Input: 128 canali, Output: 256 canali
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Riduce la dimensione spaziale (es. 12x12 -> 6x6)
        self.dropout3 = nn.Dropout(0.25)

        # Strati Completamente Connessi (Fully Connected Layers, FC)
        # Dopo i blocchi convoluzionali, l'output viene appiattito e passato a strati densi.
        # La dimensione dell'input per il primo strato FC dipende dalle dimensioni finali
        # dell'output dell'ultimo strato convoluzionale dopo il pooling.
        self.fc1 = nn.Linear(256 * 6 * 6, 256) # Primo strato FC
        self.bn7 = nn.BatchNorm1d(256) # Normalizzazione del batch per strati FC
        self.dropout4 = nn.Dropout(0.5) # Dropout più elevato per gli strati FC

        self.fc2 = nn.Linear(256, num_classes) # Strato di output, con un neurone per ogni classe di emozione

    def forward(self, x):
        """
        Definisce il passaggio in avanti (forward pass) del modello.
        Descrive come i dati fluiscono attraverso i vari strati.

        Args:
            x (torch.Tensor): Il tensore di input (immagine).

        Returns:
            torch.Tensor: Il tensore di output (logits per le classi di emozione).
        """
        # Blocco 1
        x = F.relu(self.bn1(self.conv1(x))) # Convoluzione -> Batch Norm -> ReLU
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x) # Max Pooling
        x = self.dropout1(x) # Dropout

        # Blocco 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Blocco 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flattening: Appiattisce l'output convoluzionale in un vettore 1D
        # x.size(0) è la dimensione del batch
        x = x.view(x.size(0), -1) # -1 calcola automaticamente la dimensione rimanente

        # Strati Completamente Connessi
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x) # Output finale (logits)

        return x # Restituisce i logits, che verranno poi passati a una funzione di loss (es. CrossEntropyLoss)
