import torch, torchtext
from data import get_dataloaders, Tokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
import yaml
import sys
import pandas as pd
from nltk.tokenize import word_tokenize

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

if __name__ == "__main__":

    # Read config file
    config_file = sys.argv[1]
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)

    DATA_PATH = config["data"]
    EPOCH = config["epoch"]

    # Read data and preprocess
    df = pd.read_csv(DATA_PATH, sep="|")
    data = preprocess_text_series(df["headline"]) + " " + preprocess_text_series(df["text"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Train on: {device}")
    

    # For each model, train and evaluate it
    for model_name, model_config in config["models"].items():
        print(f"\n\n> Model name: {model_name}")
        print(f"Training the following model config: \n {model_config}")

        train_loader, test_loader, NUM_CLASSES, dataset = get_dataloaders(
            file=DATA_PATH,
            tokenizer=Nltk_tok(),
            vocab_from=model_config["from"],
            device=device
        )
        VOCAB_SIZE    = len(dataset.vocab)
        rnn_config    = model_config['rnn_config']
        nn_config     = model_config['nn_config']
        embedding_dim = rnn_config['embedding_dim']

        if model_config['from'] == 'data':
            model = RNN.from_data(rnn_config=rnn_config, 
                      nn_config=nn_config, 
                      NUM_CLASSES=NUM_CLASSES,
                      vocab_size=VOCAB_SIZE,
                      embedding_dim=embedding_dim)
            
        else:
            model = RNN.from_glove(rnn_config=rnn_config,
                                nn_config=nn_config,
                                NUM_CLASSES=NUM_CLASSES,
                                glove_vectors=dataset.vectors,
                                embedding_dim=embedding_dim)
        optimizer = torch.optim.Adam(params=model.parameters())
        criterion = nn.BCELoss()

        losses = []

        
        model.to(device)

        for epoch in range(EPOCH):
            loss = train(optimizer=optimizer, criterion=criterion, model=model, train_loader=train_loader)
            print(f"epoch: {epoch}\tloss: {loss:.3f}")
            losses.append(loss)

        # Test
        performance = evaluate(model=model, criterion=criterion, test_loader=test_loader)
        print(f"Performance {model_name}: {performance}")