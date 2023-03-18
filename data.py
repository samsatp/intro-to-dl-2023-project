# Skeleton code by Chat-GPT

from typing import List, Tuple, NewType
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence

class dtype:
    indices = List[int]
    batch = Tuple[indices, indices]

class Tokenizer:
    # Base class for implementing a tokenizer
    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> List[list]:
        # Tokenize a string into a list of tokens
        raise NotImplementedError
    
def collate_fn(batch: dtype.batch) -> Tuple[PackedSequence, dtype.indices]:
    """
        the collate_fn parameter is a function that defines how to 
        collate (combine) individual data samples into batches 
        for efficient processing by a neural network.

        By default, PyTorch assumes that each sample in the dataset is 
        a tensor and combines them into a single tensor for a batch.
        However, if the samples in the dataset are of different sizes, 
        shapes or types, collate_fn is used to combine them into 
        a batch with a uniform shape and type.
    """
    xs, ys = batch

    ## Sort the sequences by their length in a descendant order
    sorted_xs = sorted(xs ,key=lambda x:len(x), reverse=True)
    lengths = [len(x) for x in sorted_xs]

    ## Pad the sequences in a batch to equal lengths
    padded_xs = pad_sequence(sorted_xs, batch_first=True)

    ## Pack the padded sequence
    packed_sequences = pack_padded_sequence(padded_xs, 
                                            lengths=lengths,
                                            batch_first=True)

    return (packed_sequences, ys)

class MultiLabelDataset(Dataset):
    def __init__(self, 
                 data:   List[str],       # List of text
                 labels: List[List[int]], # List of list of labels
                 tokenizer: Tokenizer):
        
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
        self.index2label = dict(zip([i for i in range(len(unique_labels))], unique_labels))
        self.label2index = {label: idx for idx, label in self.index2label.items()}
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
        indexed_bow = [int(idx in indexed_labels) for idx in range(self.NUM_CLASSES)]        
        label_tensor = torch.tensor(indexed_bow, dtype=torch.int16)

        return indices_tensor, label_tensor