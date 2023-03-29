import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from data import get_dataloaders

# Load datasets
train_dataset, eval_dataset, test_dataset, num_labels = get_dataloaders(file = "data/text_documents.csv", vocab_from="bert", nrows = 100000)

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Change activation to sigmoid for multi-label classification
model.classifier.activation = torch.nn.Sigmoid()

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
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

# Evaluate the model
eval_result = trainer.evaluate(eval_dataset=test_dataset)
print(eval_result)