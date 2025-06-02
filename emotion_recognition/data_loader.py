# data_loader.py
# This module is responsible for loading and pre-processing the FER-2013 dataset.

import pandas as pd # For CSV data manipulation
import numpy as np # For numerical operations, especially for images
from torch.utils.data import Dataset # Base class for PyTorch datasets
from PIL import Image # For image manipulation

class FER2013Dataset(Dataset):
    """
    Custom Dataset for the FER-2013 dataset.
    Loads data from a CSV file, extracts images and labels,
    and applies specified transformations.
    """
    def __init__(self, csv_file, phase, transform=None):
        """
        Constructor for the FER2013Dataset class.

        Args:
            csv_file (str): Path to the FER-2013 dataset CSV file.
            phase (str): Dataset phase to load ('Training', 'PublicTest', 'PrivateTest').
            transform (callable, optional): Transformations to apply to the images.
        """
        self.df = pd.read_csv(csv_file) # Loads the CSV file into a pandas DataFrame
        self.df = self.df[self.df['Usage'] == phase] # Filters the DataFrame based on the phase (Training, PublicTest, PrivateTest)
        self.transform = transform # Stores the transformations to apply

        # Maps numerical emotion labels to standard emotion names
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
        Returns the total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (image and label) from the dataset given an index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and the emotion label.
        """
        # Extract the row corresponding to the index
        row = self.df.iloc[idx]
        
        # Extract the emotion label
        emotion = row['emotion']
        
        # Extract image pixels as a string and convert them to a NumPy array
        # Pixels are a string of numbers separated by spaces
        pixels_str = row['pixels'].split(' ')
        # Converts the pixel string into a NumPy array of integers
        # and reshapes it to 48x48 pixels (FER-2013 image dimension)
        image = np.array(pixels_str, dtype='uint8').reshape(48, 48)

        # Apply transformations if specified
        if self.transform:
            # torchvision transformations expect a PIL Image or a tensor.
            # Here, we pass the NumPy array directly; torchvision.transforms.ToPILImage() will handle it.
            image = self.transform(image)

        return image, emotion # Returns the transformed image and the label