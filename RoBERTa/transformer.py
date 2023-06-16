import datasets
from datasets import Dataset
import pyarrow as pa
import pyarrow.dataset as ds


import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from data import loading_data, formatting_data, tokenizer

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def plot_loss(trainer, plot_name):
    train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    val_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    val_loss.pop()

    epochs = range(1, len(val_loss)+1)

    plt.plot(epochs, train_loss, color='blue', label='Training Loss')
    plt.plot(epochs, val_loss, color='orange', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(plot_name)


##### TASK 1 #####
print('********************* TASK 1 *********************')

# Loading and managing the data
train = loading_data('train.tsv', 'task1')
test  = loading_data('test.tsv',  'task1')

train, validation = train_test_split(train, test_size=0.2, random_state=42)
    
train = formatting_data(train)
test  = formatting_data(test)
validation = formatting_data(validation)

# Loading the model
model = RobertaForSequenceClassification.from_pretrained('projecte-aina/roberta-base-ca-v2')

# Define the training arguments 
training_args = TrainingArguments(
    output_dir='/homedtic/amoncho/CLUSTER/output/jlealtru/data_files/github/website_tutorials/results',
    evaluation_strategy = 'epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='/homedtic/amoncho/CLUSTER/output/data_files/github/website_tutorials/logs',
    logging_steps=250,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True
)

# Instantiate the trainer class
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train,
    eval_dataset=validation,
)
        
# Train the model
print('\n* TRAIN *\n')
trainer.train()
    
# Evaluate the model
print('\n* EVALUATE *\n')
trainer.evaluate(test)

# Plot loss
plot_loss(trainer, 'task1.png')
    
# Save the model and tokenizer
model_name = 'saved_roberta_model_task1'
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)

##### TASK 2 #####
print('********************* TASK 2 *********************')

# Loading and managing the data
train2 = loading_data('train.tsv', 'task2')
test2  = loading_data('test.tsv',  'task2')

train2, validation2 = train_test_split(train2, test_size=0.2, random_state=42)
    
train2 = formatting_data(train2)
test2  = formatting_data(test2)
validation2 = formatting_data(validation2)

# Loading the model
model2 = RobertaForSequenceClassification.from_pretrained('projecte-aina/roberta-base-ca-v2', num_labels=6)

# Define the training arguments 
training_args2 = TrainingArguments(
    output_dir='/homedtic/amoncho/CLUSTER/output/jlealtru/data_files/github/website_tutorials/results',
    evaluation_strategy = 'epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='/homedtic/amoncho/CLUSTER/output/data_files/github/website_tutorials/logs',
    logging_steps=250,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True
)

# Instantiate the trainer class
trainer2 = Trainer(
    model=model2,
    args=training_args2,
    compute_metrics=compute_metrics,
    train_dataset=train2,
    eval_dataset=validation2,
)
        
# Train the model
print('\n* TRAIN *\n')
trainer2.train()
    
# Evaluate the model
print('\n* EVALUATE *\n')
trainer2.evaluate(test2)

# Plot loss
plot_loss(trainer2, 'task2.png')

# Save the model and tokenizer
model_name = 'saved_roberta_model_task2'
model2.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)
