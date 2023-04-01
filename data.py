# Skeleton code by Chat-GPT

from typing import List, Tuple, NewType, Union
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from torchtext.vocab import GloVe, vocab
import pandas as pd
import json
from transformers import BertTokenizer, BertForSequenceClassification
import utils

class dtype:
    indices = torch.TensorType
    sample = Tuple[indices, indices]
    collate_input = List[sample]

class Tokenizer:
    # Base class for implementing a tokenizer
    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> List[str]:
        # Tokenize a string into a list of tokens
        return text.split()
    
def collate_fn(batch: dtype.collate_input) -> Tuple[PackedSequence, dtype.indices]:
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
    xs, ys = [sample[0] for sample in batch], [sample[1] for sample in batch]

    ## Sort the sequences by their length in a descendant order
    sorted_xs = sorted(xs ,key=lambda x:len(x), reverse=True)
    lengths = [len(x) for x in sorted_xs]

    ## Pad the sequences in a batch to equal lengths
    padded_xs = pad_sequence(sorted_xs, batch_first=True)
    return {
        "x": padded_xs,
        "y": torch.stack(ys, 0),
        "lengths": lengths
    }

class MultiLabelDataset(Dataset):
    def __init__(self, 
                 data:   List[str],       # List of text
                 labels: List[List[int]], # List of list of labels
                 tokenizer: Tokenizer,
                 device
                 ):

        # Collect the unique labels
        if labels != None:
            unique_labels = []
            for e in labels:
                unique_labels += e 
            unique_labels = set(unique_labels)
            self.index2label = dict(zip([i for i in range(len(unique_labels))], unique_labels))
            self.label2index = {label: idx for idx, label in self.index2label.items()}
            self.NUM_CLASSES = len(unique_labels)
        else:
            self.index2label = None
            self.label2index = None
            self.NUM_CLASSES = None
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = None
        self.device = device

    def get_metadata(self):
        return dict(
            index2label = self.index2label,
            label2index = self.label2index,
            NUM_CLASSES = self.NUM_CLASSES,
            VOCAB_SIZE  = self.VOCAB_SIZE
        )
        
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

        return indices_tensor.to(device=self.device), label_tensor.to(device=self.device)
    
    @classmethod
    def build_with_transformer(cls,
                               data:   List[str],       # List of text
                               labels: List[List[int]], # List of list of labels
                               tokenizer: BertTokenizer,
                               device):
        
        def __getitem__(self, idx):
            item = dict()
            # Create indices vector for headlines and texts
            encoded_texts = self.tokenizer(self.data[idx], **self.bert_tokenizer_args)
            item['input_ids'] = encoded_texts['input_ids'][0]
            item['attention_mask'] = encoded_texts['attention_mask'][0]

            # Convert the indexed labels to a PyTorch tensor
            if self.labels != None:
                indexed_labels = [self.label2index[label] for label in self.labels[idx]]
                indexed_bow = [int(idx in indexed_labels) for idx in range(self.NUM_CLASSES)]        
                label_tensor = torch.tensor(indexed_bow).float()
                item['labels'] = label_tensor
            return item
        
        cls.__getitem__ = __getitem__

        dataset = cls(data=data, labels=labels, tokenizer=tokenizer, device=device)
        dataset.vocab_by = "bert"
        dataset.bert_tokenizer_args = dict(max_length=512, 
                                           truncation=True, 
                                           padding='max_length', 
                                           return_tensors="pt")
        return dataset
    
    @classmethod
    def build_vocab_from_data(cls, 
                              data:   List[str],       # List of text
                              labels: List[List[int]], # List of list of labels
                              tokenizer: Tokenizer,
                              device):
        
        dataset = cls(data=data, labels=labels, tokenizer=tokenizer, device=device)
        dataset.vocab_by = "data"
        dataset.vocab = torchtext.vocab.build_vocab_from_iterator(
                            data, 
                            specials=['<PAD>', '<UNK>'], 
                            special_first=True,
                            min_freq=100
                        )
        dataset.vocab.set_default_index(dataset.vocab['<UNK>'])
        dataset.VOCAB_SIZE = len(dataset.vocab)
        return dataset
    
    @classmethod
    def build_vocab_from_pretrain_emb(cls, 
                                      data:   List[str],       # List of text
                                      labels: List[List[int]], # List of list of labels
                                      tokenizer: Tokenizer,
                                      pretrained_name: str,
                                      device):
        
        dataset = cls(data=data, labels=labels, tokenizer=tokenizer, device=device)
        dataset.vocab_by = pretrained_name
        pretrained_emb = torchtext.vocab.pretrained_aliases[pretrained_name]()
        dataset.vectors = pretrained_emb.vectors
        dataset.vocab = torchtext.vocab.vocab(pretrained_emb.stoi)
        dataset.vocab.insert_token("<UNK>",0)
        dataset.vocab.insert_token("<PAD>",1)
        dataset.vocab.set_default_index(0)
        dataset.vectors = torch.cat((torch.zeros(1, dataset.vectors.shape[1]), dataset.vectors))  # Zero vector for <UNK> token
        return dataset
    
    
def get_dataloaders(file, 
                    device,
                    vocab_from: str = "data",
                    tokenizer: Tokenizer = None,
                    nrows = None, 
                    train_size = 0.8, 
                    batch_size = 64):
    '''
    Get train and test dataset loaders with headlines, texts and labels.

    # Parameters
    ---
    - `vocab_from`: default = "data"
        - "data": build the vocab from the data
        - "bert": use bert vocab, tokenizer, and embedding
        - one of the `torchtext.vocab.pretrained_aliases` key: use its vocab and embedding
    - `tokenizer`: Optional, default = None
        - not required, if `vocab_from = bert`
        - otherwise, required to pass an object of the Tokenizer class
    '''
    # Load the data
    data, labels = utils.get_data(file, nrows)
    
    if vocab_from == "data":
        dataset = MultiLabelDataset.build_vocab_from_data(data, labels, tokenizer = tokenizer, device=device)
        get_collate_fn = lambda: collate_fn
    elif vocab_from == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = MultiLabelDataset.build_with_transformer(data, labels, tokenizer = tokenizer, device=device)
        get_collate_fn = lambda: None
    else:
        dataset = MultiLabelDataset.build_vocab_from_pretrain_emb(data, labels, tokenizer = tokenizer, pretrained_name=vocab_from, device=device)
        get_collate_fn = lambda: collate_fn


    # Split the dataset into training and test data
    train_num = int(train_size * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_num, len(dataset) - train_num])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=get_collate_fn())
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=get_collate_fn())
    num_classes = dataset.NUM_CLASSES
    return train_loader, test_loader, num_classes, dataset

def get_datasets(file, 
                device,
                vocab_from: str = "data",
                tokenizer: Tokenizer = None,
                nrows = None, 
                train_size = 0.6,
                val_size = 0.2):
    '''
    Same as get_dataloaders but returns the actual datasets and also a validation set.
    '''
    # Load the data
    data, labels = utils.get_data(file, nrows)
    
    if vocab_from == "data":
        dataset = MultiLabelDataset.build_vocab_from_data(data, labels, tokenizer = tokenizer, device=device)
        get_collate_fn = lambda: collate_fn
    elif vocab_from == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = MultiLabelDataset.build_with_transformer(data, labels, tokenizer = tokenizer, device=device)
        get_collate_fn = lambda: None
    else:
        dataset = MultiLabelDataset.build_vocab_from_pretrain_emb(data, labels, tokenizer = tokenizer, pretrained_name=vocab_from, device=device)
        get_collate_fn = lambda: collate_fn


    # Split the dataset into training and test data
    train_num = int(train_size * len(dataset))
    val_num = int(val_size * len(dataset))
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_num, val_num, len(dataset) - train_num - val_num])
    num_classes = dataset.NUM_CLASSES
    return train_dataset, val_dataset, test_dataset, num_classes, dataset
