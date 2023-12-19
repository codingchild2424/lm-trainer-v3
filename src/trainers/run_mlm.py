import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import argparse

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_name_or_path', type=str, default=None)
    p.add_argument('--train_file_path', type=str, default=None)
    p.add_argument('--test_file_path', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--text_column_name', type=str, default='text')
    p.add_argument('--label_column_name', type=str, default='label')
    p.add_argument('--num_labels', type=int, default=2)
    p.add_argument('--id2label', type=dict, default={ 0: "CLEAN", 1: "UNCLEAN" })
    p.add_argument('--per_device_train_batch_size', type=int, default=8)
    p.add_argument('--per_device_eval_batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=3)


    config = p.parse_args()

    return config

def main(config):
    # 1. Load and Preprocess the Data
    # config.train_file_path endwith .csv
    if config.train_file_path.endswith('.csv'):
        data = pd.read_csv(config.train_file_path, sep=",", header=0)
    elif config.train_file_path.endswith('.tsv'):
        data = pd.read_csv(config.train_file_path, sep="\t", header=0)
    else:
        raise ValueError("train_file_path must end with .csv or .tsv")
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_texts, train_labels = train['text'].tolist(), train['label'].tolist()
    test_texts, test_labels = test['text'].tolist(), test['label'].tolist()

    # 2. Tokenize the Data using AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

    # 3. Convert Data into a Torch Dataset
    class CleanUncleanDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CleanUncleanDataset(train_encodings, [0 if label == "CLEAN" else 1 for label in train_labels])
    test_dataset = CleanUncleanDataset(test_encodings, [0 if label == "CLEAN" else 1 for label in test_labels])

    # 4. Initialize the Model and Train using AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path, 
        num_labels=config.num_labels,
        id2label=config.id2label,
        )

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        do_train=True,
        do_eval=True,
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    trainer.save_model(config.output_dir)
    # save tokenizer
    tokenizer.save_pretrained(config.output_dir)

    # 5. Evaluate the Model
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    config = define_argparser()
    main(config)