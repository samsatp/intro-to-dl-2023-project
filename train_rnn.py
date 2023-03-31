import torch, torchtext
from data import get_dataloaders, Tokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
import yaml
import sys
import pandas as pd
from nltk.tokenize import word_tokenize
import warnings

from model import *
from data import *
from utils import *

all_models = ["data"] + list(torchtext.vocab.pretrained_aliases.keys()) 

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
    LOSS_THRESH = 0.001
    EARLY_STOP_AT = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Train on: {device}")
    
    perf_json = {}

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

        try:
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
        except Exception as e:
            warnings.warn(f"Model built failed: { model_name } \n {model_config}")
            continue
        optimizer = torch.optim.Adam(params=model.parameters())
        criterion = nn.BCELoss()

        losses = []
        early_stop_counter = 0
        
        model.to(device)

        for epoch in range(EPOCH):
            loss = train(optimizer=optimizer, criterion=criterion, model=model, train_loader=train_loader)
            print(f"epoch: {epoch}\tloss: {loss:.3f}")
            if len(losses) > 0 and losses[-1] - loss <= LOSS_THRESH:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
            if early_stop_counter == EARLY_STOP_AT:
                break
            losses.append(loss)

        # Test
        performance = evaluate(model=model, criterion=criterion, test_loader=test_loader)
        print(f"Performance {model_name}: {performance}")

        perf_json[model_name] = performance

        torch.save(model, f"{model_name}.model")

    print(perf_json)