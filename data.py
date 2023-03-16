# Skeleton code by Chat-GPT

from typing import List
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    # Base class for implementing a tokenizer
    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> List[list]:
        # Tokenize a string into a list of tokens
        raise NotImplementedError

class MultiLabelDataset(Dataset):
    def __init__(self, 
                 data:   List[List[str]], 
                 labels: List[List[int]], 
                 tokenizer):
        
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            data, 
            specials=['<PAD>', '<UNK>'], 
            special_first=True
        )
        self.vocab.set_default_index(self.vocab['<UNK>'])

        unique_labels = []
        for e in labels:
            unique_labels += e 
            
        unique_labels = set(unique_labels)
        self.label2index = dict(zip(unique_labels, [i for i in range(len(unique_labels))]))
        self.NUM_CLASSES = len(labels[0])
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Create indices vector
        encoded_text = self.vocab(self.tokenizer(self.data[idx]))
        indices_tensor = torch.tensor(encoded_text)
        
        # Convert the indexed labels to a PyTorch tensor
        indexed_labels = [self.label2index[label] for label in self.labels[idx]]
        label_tensor = torch.tensor(indexed_labels, dtype=torch.int16)

        return indices_tensor, label_tensor