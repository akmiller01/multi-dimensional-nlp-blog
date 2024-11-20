from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import torch
import evaluate
import numpy as np
from bert_multi_model import BertForMultiSequenceClassification
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
login(token=HF_TOKEN)

card = "alex-miller/ODABert"
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


dataset = load_dataset("alex-miller/iati-disability-structured")

unique_labels = [
    "Significant disability objective",
    "Principal disability objective"
]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}

def preprocess_function(example):
    label = example['disability_sig']
    labels = [0. for i in range(len(unique_labels))]
    if label == 1:
        labels[label2id["Significant disability objective"]] = 1.
    elif label == 2:
        labels[label2id["Significant disability objective"]] = 1.
        labels[label2id["Principal disability objective"]] = 1.

    result_dict = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'labels': labels
    }
    text_dimensions = ['title_narrative', 'description_narrative', 'transaction_description_narrative']
    for text_dimension in text_dimensions:
        text = example[text_dimension] if example[text_dimension] is not None else ""
        tokenized_dimension = tokenizer(text, truncation=True)
        for tokenizer_key in ['input_ids', 'token_type_ids', 'attention_mask']:
            result_dict[tokenizer_key] += tokenized_dimension[tokenizer_key]

    return result_dict

dataset = dataset.map(preprocess_function, remove_columns=[
    'disability_sig', 'text', 'languages', 'title_narrative', 'description_narrative', 'transaction_description_narrative'
])

weight_list = list()
total_rows = dataset['train'].num_rows + dataset['test'].num_rows
print("Weights:")
for label in unique_labels:
    label_idx = label2id[label]
    positive_filtered_dataset = dataset.filter(lambda example: example['labels'][label_idx] == 1.)
    negative_filtered_dataset = dataset.filter(lambda example: example['labels'][label_idx] == 0.)
    pos_label_rows = positive_filtered_dataset['train'].num_rows + positive_filtered_dataset['test'].num_rows
    neg_label_rows = negative_filtered_dataset['train'].num_rows + negative_filtered_dataset['test'].num_rows
    label_weight = neg_label_rows / pos_label_rows
    weight_list.append(label_weight)
    print("{}: {}".format(label, label_weight))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
weights = torch.tensor(weight_list)
weights = weights.to(device)

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
   return 1/(1 + np.exp(-x))


def compute_metrics(eval_pred):
   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


model = BertForMultiSequenceClassification.from_pretrained(
    card,
    num_labels=len(id2label.keys()), 
    id2label=id2label,
    label2id=label2id, 
    problem_type="multi_label_classification"
)
model.class_weights = weights

training_args = TrainingArguments(
    'multi-dimensional-disability',
    learning_rate=1e-6,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=True,
    save_total_limit=5,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()