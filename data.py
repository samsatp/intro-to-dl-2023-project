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
                 tokenizer: Tokenizer
                 ):

        # Collect the unique labels
        unique_labels = []
        for e in labels:
            unique_labels += e 
        unique_labels = set(unique_labels)
        self.index2label = dict(zip([i for i in range(len(unique_labels))], unique_labels))
        self.label2index = {label: idx for idx, label in self.index2label.items()}
        self.NUM_CLASSES = len(unique_labels)
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
    
    @classmethod
    def build_with_transformer(cls,
                               data:   List[str],       # List of text
                               labels: List[List[int]], # List of list of labels
                               tokenizer: BertTokenizer):
        
        def __getitem__(self, idx):
            item = dict()
            # Create indices vector for headlines and texts
            encoded_texts = self.tokenizer(self.data[idx], **self.bert_tokenizer_args)
            item['texts'] = encoded_texts['input_ids'][0]
            item['attention_mask'] = encoded_texts['attention_mask'][0]

            # Convert the indexed labels to a PyTorch tensor
            indexed_labels = [self.label2index[label] for label in self.labels[idx]]
            indexed_bow = [int(idx in indexed_labels) for idx in range(self.NUM_CLASSES)]        
            label_tensor = torch.tensor(indexed_bow).float()
            item['labels'] = label_tensor
            return item
        
        cls.__getitem__ = __getitem__

        dataset = cls(data=data, labels=labels, tokenizer=tokenizer)
        dataset.vocab_by = "bert"
        dataset.bert_tokenizer_args = dict(max_length=32, 
                                           truncation=True, 
                                           padding=True, 
                                           return_tensors="pt")
        return dataset
    
    @classmethod
    def build_vocab_from_data(cls, 
                              data:   List[str],       # List of text
                              labels: List[List[int]], # List of list of labels
                              tokenizer: Tokenizer):
        
        dataset = cls(data=data, labels=labels, tokenizer=tokenizer)
        dataset.vocab_by = "data"
        dataset.vocab = torchtext.vocab.build_vocab_from_iterator(
                            data, 
                            specials=['<PAD>', '<UNK>'], 
                            special_first=True
                        )
        dataset.vocab.set_default_index(dataset.vocab['<UNK>'])
        return dataset
    
    @classmethod
    def build_vocab_from_pretrain_emb(cls, 
                                      data:   List[str],       # List of text
                                      labels: List[List[int]], # List of list of labels
                                      tokenizer: Tokenizer,
                                      pretrained_name: str):
        
        dataset = cls(data=data, labels=labels, tokenizer=tokenizer)
        dataset.vocab_by = pretrained_name
        pretrained_emb = torchtext.vocab.pretrained_aliases[pretrained_name]()
        dataset.vectors = pretrained_emb.vectors
        dataset.vocab = torchtext.vocab.vocab(pretrained_emb.stoi)
        dataset.vocab.insert_token("<UNK>",0)
        dataset.vocab.insert_token("<PAD>",1)
        dataset.vocab.set_default_index(0)
        return dataset
    
    
def get_dataloaders(file, 
                    vocab_from: str = "data",
                    tokenizer: Union[str, Tokenizer] = "",
                    nrows = None, 
                    train_size = 0.8, 
                    batch_size = 64):
    '''
    Get train and test dataset loaders with headlines, texts and labels.
    '''
    def get_data(file, nrows):
        # Load headlines, texts and labels
        df = pd.read_csv(file, sep = '|', nrows=nrows)
        df["headline"].fillna("", inplace=True)
        df["text"].fillna("", inplace=True)

        # Select the desired number of rows or all rows
        data = df["headline"].str.strip() + " " + df["text"].str.strip()
        labels = df['label'].values
        labels = [json.loads(item.replace("'", "\"")) for item in labels]

        return data, labels

    # Load the data
    data, labels = get_data(file, nrows)
        
    if isinstance(tokenizer, Tokenizer):
        if vocab_from == "data":
            dataset = MultiLabelDataset.build_vocab_from_data(data, labels, tokenizer = tokenizer)
        else:
            dataset = MultiLabelDataset.build_vocab_from_pretrain_emb(data, labels, tokenizer = tokenizer, pretrained_name=vocab_from)
        get_collate_fn = lambda: collate_fn

    elif tokenizer.lower() == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = MultiLabelDataset.build_with_transformer(data, labels, tokenizer = tokenizer)
        get_collate_fn = lambda: None
    else:
        raise NotImplementedError("Tokenizer not implemented")

    # Split the dataset into training and test data
    train_num = int(train_size * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_num, len(dataset) - train_num])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=get_collate_fn())
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=get_collate_fn())
    num_classes = dataset.NUM_CLASSES
    return train_loader, test_loader, num_classes
