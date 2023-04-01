import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoConfig, AutoModel
from data import get_dataloaders
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# Load datasets
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset, eval_dataset, test_dataset, num_labels, dataset = get_dataloaders(file = "data/text_documents.csv", vocab_from="bert", nrows = 100000, device = device)

# Save labels
pd.DataFrame(list(dataset.label2index.items()), columns=['label', 'index']).to_csv("label2index.csv", index = False)

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Change activation to sigmoid for multi-label classification
model.classifier.activation = torch.nn.Sigmoid()

# Define training arguments
training_args = TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    evaluation_strategy='steps',
    eval_steps=500,
    save_total_limit=1,
    gradient_accumulation_steps=2,
    save_steps = 500
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

print("Model training finished")

def load_model(config_path, model_path):
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModel.from_pretrained(model_path, config=config)
    return model

# Evaluate the model
#model = load_model("model-checkpoint/config.json", "model-checkpoint/pytorch_model.bin")

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

with torch.no_grad():
    outputs = []
    labels = []
    for i, batch in enumerate(test_loader):
        input = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels.append(batch["labels"].to(device))
        output = model(input, attention_mask = attention_mask).logits.to(device)
        outputs.append(output)
outputs, labels = torch.cat(outputs, dim = 0), torch.cat(labels, dim = 0)


def evaluate_model(decision_threshold):
    y_trues = labels.cpu()
    y_preds = (outputs >= decision_threshold).float().cpu()
    precisions_micro = precision_score(y_trues, y_preds, average='micro')
    precisions_macro = precision_score(y_trues, y_preds, average='macro')
    recalls_micro = recall_score(y_trues, y_preds, average='micro')
    recalls_macro = recall_score(y_trues, y_preds, average='macro')
    f1s_micro = f1_score(y_trues, y_preds, average='micro')
    f1s_macro = f1_score(y_trues, y_preds, average='macro')

    return dict(
        decision_threshold = decision_threshold,
        precision_micro_avg = precisions_micro,
        precision_macro_avg = precisions_macro,
        recall_micro_avg = recalls_micro,
        recall_macro_avg = recalls_macro,
        f1_micro_avg = f1s_micro,
        f1_macro_avg = f1s_macro
    )

results = []
for x in np.linspace(0.1, 0.5, 5):
    result = evaluate_model(x)
    results.append(result)

result_path = "results.csv"
pd.DataFrame(results).to_csv(result_path, index = False)
print("Results saved to", result_path)

model_save_path = "bert_model/model"
tokenizer_save_path = "bert_model/tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)
print("model and tokenizer saved to", model_save_path, "and", tokenizer_save_path)

