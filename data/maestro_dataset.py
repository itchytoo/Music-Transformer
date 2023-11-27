from torch.utils.data import Dataset
import torch
from miditok import REMI
import os 

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
        
        # if it is shorter than self.sequence_length, then pad the sequence with zeros and return it
        if sequence.shape[0] < self.sequence_length:
            return torch.cat((sequence, torch.zeros(self.sequence_length - sequence.shape[0])))  

        # otherwise, return a random window of length self.sequence_length from the sequence    
        window_start = torch.randint(0, sequence.shape[0] - self.sequence_length, (1,))
        window_end = window_start + self.sequence_length
        return sequence[window_start:window_end]

    def __len__(self):
        return len(self.token_paths)