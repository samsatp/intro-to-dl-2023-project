# Skeleton code by Chat-GPT

import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List
from data import MultiLabelDataset, collate_fn, Tokenizer
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence


    
class MultiLabelClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, 
                                                lengths=lengths,
                                                batch_first=True)
        lstm_output, _ = self.lstm(packed_embedded)
        lstm_unpacked, len_unpacked = pad_packed_sequence(lstm_output, batch_first=True)

        max_pool, _ = torch.max(lstm_unpacked, dim=1)
        out = self.fc(max_pool)
        out = torch.sigmoid(out)
        return out

def train(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    for batch_idx, inputs in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(inputs['x'], inputs['lengths'])
        loss = criterion(output, inputs['y'].float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, criterion, test_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target.float())
            running_loss += loss.item()
    return running_loss / len(test_loader)

# Example usage:
if __name__ == '__main__':
    import utils
    import glob, os
    
    # Define the hyperparameters
    BATCH_SIZE = 32
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    # Load the data
    #files = glob.glob(os.path.join("data","*.xml"))
    #print(f"There are {len(files)} files")
    #headlines, texts, labels = utils.parse_xml(files=files)
    #X = [(a+" "+b).lower() for a,b in zip(headlines, texts)]
    # TODO: get_data_from_text_files(): read data from saved text files instead
    
    X, y = utils.get_data_from_text_files("data/extracted/headlines.txt", "data/extracted/texts.txt", "data/extracted/labels.txt")
    print("Read done")
    print("\tdata size: ", len(X))

    # Create a tokenizer
    class TrivialTokenizer(Tokenizer):
        def __call__(self, text: str) -> List[list]:
            return text.split()
    tokenizer = TrivialTokenizer()

    # Build dataset & dataloader
    train_dataset = MultiLabelDataset(X, y, tokenizer)
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              collate_fn=collate_fn)
    
    NUM_CLASSES = train_dataset.NUM_CLASSES
    VOCAB_SIZE = len(train_dataset.vocab)
    print("\tNUM_CLASSES: ", NUM_CLASSES)
    print("\tVOCAB_SIZE: ", VOCAB_SIZE)

    # Create the model
    model = MultiLabelClassifier(vocab_size=VOCAB_SIZE, 
                                 embedding_dim=EMBEDDING_DIM, 
                                 hidden_dim=HIDDEN_DIM, 
                                 num_classes=NUM_CLASSES)
    
    # Create the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    print("Start training")
    # Train the model
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, optimizer, criterion, train_loader)
        print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, train_loss))
