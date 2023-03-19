# Skeleton code by Chat-GPT

from typing import List, Tuple, NewType
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
import pandas as pd
import numpy as np
import json
from transformers import BertTokenizer, BertForSequenceClassification

class dtype:
    indices = torch.TensorType
    sample = Tuple[indices, indices]
    collate_input = List[sample]

class Tokenizer:
    # Base class for implementing a tokenizer
    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> dtype.indices:
        # Tokenize a string into a list of tokens
        raise NotImplementedError

class MultiLabelDataset(Dataset):
    def __init__(self, 
                 headlines: List[str], # List of headlines
                 texts: List[str],       # List of text
                 labels: List[List[int]], # List of list of labels
                 tokenizer: Tokenizer):
        
        unique_labels = []
        for e in labels:
            unique_labels += e 
            
        unique_labels = set(unique_labels)

        self.index2label = dict(zip([i for i in range(len(unique_labels))], unique_labels))
        self.label2index = {label: idx for idx, label in self.index2label.items()}
        self.NUM_CLASSES = len(unique_labels)
        self.headlines = headlines
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.encoded_headlines = self.tokenizer(self.headlines, max_length=32, truncation=True, padding=True, return_tensors="pt")
        self.encoded_texts = self.tokenizer(self.texts, max_length=512, truncation=True, padding=True, return_tensors="pt")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = dict()
        # Create indices vector for headlines and texts
        
        item['headlines'] = self.encoded_headlines['input_ids'][idx]
        item['texts'] = self.encoded_texts['input_ids'][idx]
        item['attention_mask'] = self.encoded_texts['attention_mask'][idx]

        # Convert the indexed labels to a PyTorch tensor
        indexed_labels = [self.label2index[label] for label in self.labels[idx]]
        indexed_bow = [int(idx in indexed_labels) for idx in range(self.NUM_CLASSES)]        
        label_tensor = torch.tensor(indexed_bow).float()
        item['labels'] = label_tensor

        return item
    
def get_dataloaders(file, num_rows = None, train_size = 0.8, batch_size = 64):
    '''
    Get train and test dataset loaders with headlines, texts and labels.
    '''
    def get_data(file, num_rows):
        # Load headlines, texts and labels
        df = pd.read_csv(file, sep = '|')
        df["headline"].fillna("", inplace=True)
        df["text"].fillna("", inplace=True)

        # Select the desired number of rows or all rows
        if num_rows == None:
            headlines = df['headline'].values.tolist()
            texts = df['text'].values.tolist()
            labels = df['label'].values
            labels = [json.loads(item.replace("'", "\"")) for item in labels]
        else:
            headlines = df['headline'][0:num_rows].values.tolist()
            texts = df['text'][0:num_rows].values.tolist()
            labels = df['label'][0:num_rows].values
            labels = [json.loads(item.replace("'", "\"")) for item in labels]
        return headlines, texts, labels

    # Load the data
    headlines, texts, labels = get_data(file, num_rows)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create a dataset that contains headlines, texts and labels
    dataset = MultiLabelDataset(headlines, texts, labels, tokenizer = tokenizer)

    # Split the dataset into training and test data
    train_num = int(train_size * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_num, len(dataset) - train_num])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    num_classes = dataset.NUM_CLASSES
    return train_loader, test_loader, num_classes