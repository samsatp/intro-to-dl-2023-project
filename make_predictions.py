import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from data import MultiLabelDataset
import utils

def load_pretrained(model_path):
    # Get the number of labels
    num_labels = len(pd.read_csv("label2index.csv"))

    # Load model state
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.classifier.activation = torch.nn.Sigmoid()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

# Get the pretrained model
model = load_pretrained("models/bert_model/pytorch_model.bin")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load test data
test_data_path = "data/test_data.csv"
data, labels = utils.get_data(file=test_data_path, nrows = None)
test_dataset = MultiLabelDataset.build_with_transformer(data, labels, tokenizer = BertTokenizer.from_pretrained("bert-base-uncased"))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Calculate outputs
outputs = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        input = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output = model(input, attention_mask = attention_mask).logits
        outputs.append(output)
    outputs = torch.cat(outputs, dim = 0)

def save_predictions(outputs):
    # Predict outputs based on 0.3 threshold.
    outputs = outputs.cpu()
    y_preds = (outputs >= 0.3).float()

    # Get the labels as columns
    df = pd.DataFrame(y_preds.numpy().astype(int))
    label2index = pd.read_csv("label2index.csv")
    df.columns = label2index['label'].values

    # Get the filenames as index
    df.index = pd.read_csv(test_data_path, sep = "|").index
    df.index.name = 'Filename'

    # Save to a file
    df.to_csv("predictions.tsv", sep = "\t")
    df.to_csv("predictions.csv")

save_predictions(outputs)