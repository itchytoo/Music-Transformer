from torch.utils.data import Dataset
import torch
from miditok import REMI
import os 
import pandas as pd

class MaestroDataset(Dataset):
    """Maestro MIDI (tokenized) Dataset"""

    def __init__(self, root_dir, tokenizer, sequence_length):
        """
        Arguments:
            root_dir (string): path to the root directory containing the tokenized MIDI files
        """
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        # get the paths to all the tokenized MIDI files in the MAESTRO dataset
        self.token_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                #skip the non-JSON files
                if file[-5:] != '.json':
                    continue
                file_path = os.path.join(root, file)
                self.token_paths.append(file_path)
        
    def __getitem__(self, index):
        """
        Stochastically return a window of length self.sequence_length from the training sequence at the given index.
        The window is returned as a tensor of shape (self.sequence_length). The window is chosen uniformly at random.

        Arguments:
            index (int): index of the item to be returned
        """
        # load the tokenized MIDI file and return the tokens as a tensor
        sequence = torch.Tensor(self.tokenizer.load_tokens(self.token_paths[index])['ids'])
        
        # if it is shorter than self.sequence_length, then pad the sequence with zeros at the start and return it
        if sequence.shape[0] < self.sequence_length:
            return torch.cat((torch.zeros(self.sequence_length - sequence.shape[0], sequence)))  

        # otherwise, return a random window of length self.sequence_length from the sequence    
        window_start = torch.randint(0, sequence.shape[0] - self.sequence_length, (1,))
        window_end = window_start + self.sequence_length
        return sequence[window_start:window_end]

    def __len__(self):
        return len(self.token_paths)

class LakhDataset(Dataset):
    """Lakh MIDI (tokenized) Dataset"""

    def __init__(self, root_dir, label_path, sequence_length):
        """
        Arguments:
            root_dir (string): path to the root directory containing the tokenized MIDI files
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.genre_labels = pd.read_csv(label_path)
        self.genre_to_index = {'Pop_Rock': 0, 'Latin': 1, 'Vocal': 2, 'RnB': 3, 
                               'Folk': 4, 'Jazz': 5, 'Reggae': 6, 'Blues': 7, 'New Age': 8, 
                               'International': 9, 'Country': 10, 'Rap': 11, 'Electronic': 12}

        # get the paths to all the tokenized MIDI files in the Lakh dataset
        self.token_paths = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                hash = file.split('.')[0]
                # skip the MIDI files that do not have a genre label
                if hash not in self.genre_labels['Hash'].values:
                    continue
                file_path = os.path.join(root, file)
                self.token_paths.append(file_path)

    def __getitem__(self, index):
        """
        Stochastically return a window of length self.sequence_length from the training sequence at the given index.
        The window is returned as a tensor of shape (self.sequence_length). The window is chosen uniformly at random.

        Arguments:
            index (int): index of the item to be returned
        """
        # retrieve the genre label for the current MIDI file
        hash = os.path.basename(self.token_paths[index]).split('.')[0]
        genre = self.genre_labels.loc[self.genre_labels['Hash'] == hash]['Genre'].values[0]
        #genre_index = self.genre_to_index[genre]
        genre_index=1

        # load the tokenized MIDI file and return the tokens as a tensor
        with open(self.token_paths[index]) as f:
            sequence = torch.Tensor([int(token) for token in f.read().split()])
        
        # if it is shorter than self.sequence_length, then pad the sequence with zeros and return it
        if sequence.shape[0] < self.sequence_length:
            return torch.cat((sequence, torch.zeros(self.sequence_length - sequence.shape[0])))  

        # otherwise, return a random window of length self.sequence_length from the sequence    
        window_start = torch.randint(0, sequence.shape[0] - self.sequence_length, (1,))
        window_end = window_start + self.sequence_length

        return (sequence[window_start:window_end], genre_index)

    def __len__(self):
        return len(self.token_paths)
        