from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import torch.nn.functional as F


def prepare_dataset(texts, labels):
    data = {"text": texts, "label": labels}
    dataset = Dataset.from_dict(data)
    return dataset

def tokenize_dataset(dataset, tokenizer):
    def preprocess(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_token_type_ids=True
        )

    tokenized_dataset = dataset.map(preprocess, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    return tokenized_dataset


def build_lora_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(model, config)
    return model, tokenizer

def train_model(model, tokenizer, tokenized_dataset): 
    training_args = TrainingArguments(
        output_dir="./result",
        per_device_train_batch_size=2,
        num_train_epochs=10,
        no_cuda=True,
        logging_dir="./logs",
        logging_steps=1,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    return model

def predict_sentiment(model, tokenizer, text, return_probs=False):
    inputs = tokenizer([text], return_tensors='pt', truncation=True, padding="max_length", max_length=64, return_token_type_ids=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)[0]  # Extract from batch

    predicted_class = torch.argmax(probs).item()
    confidence = torch.max(probs).item()
    sentiment = "Positive" if predicted_class == 1 else "Negative"

    if return_probs:
        return sentiment, confidence, probs.tolist()
    return sentiment, confidence
