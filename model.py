# Skeleton code by Chat-GPT

import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List
from data import MultiLabelDataset


    
class MultiLabelClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)
        max_pool, _ = torch.max(lstm_output, dim=1)
        out = self.fc(max_pool)
        out = torch.sigmoid(out)
        return out

def train(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float())
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
    # Define the hyperparameters
    BATCH_SIZE = 32
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    
    NUM_EPOCHS = 10
    
    # Create the tokenizer
    tokenizer = lambda x: x.split()
    
    # Load the data
    train_data = ["This is a positive example", "This is a negative example", "This is a neutral example"]
    train_labels = [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    train_dataset = MultiLabelDataset(train_data, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    NUM_CLASSES = train_dataset.NUM_CLASSES
    print("NUM_CLASSES =", NUM_CLASSES)
    
    # Create the model
    model = MultiLabelClassifier(vocab_size=1000, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
    
    # Create the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # Train the model
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, optimizer, criterion, train_loader)
        print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, train_loss))
    
