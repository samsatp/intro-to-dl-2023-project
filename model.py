# Skeleton code by Chat-GPT

import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from data import MultiLabelDataset, collate_fn, Tokenizer
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

class RNN(nn.Module):
    def __init__(self, rnn_config, nn_config, NUM_CLASSES):
        super(RNN, self).__init__()
        self.embedding = None
        self.rnn_config = rnn_config
        self.rnn = None
        self.fc = torch.nn.Linear(out_features=NUM_CLASSES, **nn_config)

    def _set_RNN(self, emb_dim):
        self.rnn_config['params']['input_size'] = emb_dim

        if self.rnn_config['type'] == 'lstm':
            self.rnn = torch.nn.LSTM(**self.rnn_config['params'])
        elif self.rnn_config['type'] == 'gru':
            self.rnn = torch.nn.GRU(**self.rnn_config['params'])
        else:
            self.rnn = torch.nn.RNN(**self.rnn_config['params'])

    @classmethod
    def from_data(cls, rnn_config, nn_config, NUM_CLASSES, vocab_size, embedding_dim):
        model = cls(rnn_config, nn_config, NUM_CLASSES)
        model.embedding = nn.Embedding(vocab_size, embedding_dim)
        model._set_RNN(emb_dim=embedding_dim)
        return model

    @classmethod
    def from_glove(cls, rnn_config, nn_config, NUM_CLASSES, glove_vectors, embedding_dim):
        model = cls(rnn_config, nn_config, NUM_CLASSES)
        model.embedding = torch.nn.Embedding.from_pretrained(embeddings=glove_vectors, freeze=True)
        model._set_RNN(emb_dim=embedding_dim)
        return model
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, 
                                                lengths=lengths,
                                                batch_first=True)
        lstm_output, _ = self.rnn(packed_embedded)
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
    accuracies = []
    y_preds = []   # For binary predictions of all batches in the test set
    y_trues = []   # For binary true values of all batches in the test set

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs = batch['x']
            target = batch['y']
            lengths = batch['lengths']
            output = model(inputs, lengths)

            # Loss
            loss = criterion(output, target.float())
            running_loss += loss.item()
            
            # Accuracy
            predictions = (output >= 0.5).float()
            accuracy = torch.sum(predictions == target).float() / predictions.numel()
            accuracies.append(accuracy)

            y_pred_binary = predictions.int()
            y_preds.append(y_pred_binary)
            y_trues.append(target)
    
    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)

    test_loss = np.mean(running_loss)
    test_acc = np.mean(accuracies)
    precisions_micro = precision_score(y_trues, y_preds, average='micro')
    precisions_macro = precision_score(y_trues, y_preds, average='macro')
    recalls_micro = recall_score(y_trues, y_preds, average='micro')
    recalls_macro = recall_score(y_trues, y_preds, average='macro')
    f1s_micro = f1_score(y_trues, y_preds, average='micro')
    f1s_macro = f1_score(y_trues, y_preds, average='macro')
    return dict(
        loss = test_loss,
        acc = test_acc,
        precision_micro_avg = precisions_micro,
        precision_macro_avg = precisions_macro,
        recall_micro_avg = recalls_micro,
        recall_macro_avg = recalls_macro,
        f1_micro_avg = f1s_micro,
        f1_macro_avg = f1s_macro
    )
