import torch, torchtext
from data import get_dataloaders, Tokenizer
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from model import *
from data import *
from utils import *

all_models = ["data"] + list(torchtext.vocab.pretrained_aliases.keys()) 

def preprocess_text_series(text: pd.Series):
    text = text.str.lower()
    text = text.str.strip()
    return text

class Nltk_tok(Tokenizer):
    def __call__(self, text: str) -> List[str]:
        return word_tokenize(text)
    
def evaluate(test_loader, criterion):
    accuracies = []
    running_loss = 0
    y_preds = []   # For binary predictions of all batches in the test set
    y_trues = []   # For binary true values of all batches in the test set

    for batch_idx, batch in enumerate(test_loader):
        target = batch['y']
        output = torch.tensor(np.ones_like(target)).float()

        # Loss
        loss = criterion(output, target.float())
        running_loss += loss.item()
        
        # Accuracy
        predictions = (output >= 0.5).float()
        accuracy = torch.sum(predictions == target).float() / predictions.numel()
        accuracies.append(accuracy.cpu())

        y_pred_binary = predictions.int()
        y_preds.append(y_pred_binary.cpu())
        y_trues.append(target.cpu())

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)

    test_loss = np.median(running_loss)
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

if __name__ == "__main__":

    DATA_PATH = "data/data.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, NUM_CLASSES, dataset = get_dataloaders(
        file=DATA_PATH,
        tokenizer=Nltk_tok(),
        vocab_from="data",
        device=device
    )
    criterion = nn.BCELoss()
    perf = evaluate(test_loader=test_loader, criterion=criterion)
    print(perf)

    

