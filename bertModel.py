from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

class BertModel:
    checkpoint = 'bert-base-uncased'

    def __init__(self, datasets):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=4)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.datasets = self.datasets
    
    def tokenize_function(self, text):
        return self.tokenizer(text, truncation=True)
    
    def process_ds(self):
        # Tokenize the datasets
        tokenized_ds = self.datasets.map(self.tokenize_function, batched=True)

        # Rename the columns
        tokenized_ds.rename_column_("label", "labels")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        print(tokenized_ds)
        
        # Split the dataset
        tokenized_ds = tokenized_ds.train_test_split(test_size=0.2)
        tokenized_train_dataset = tokenized_ds["train"]
        tokenized_test_dataset = tokenized_ds["test"]
        return tokenized_train_dataset, tokenized_test_dataset

    def train(self, datasets):
        # Tokenize the datasets
        tokenized_ds = self.datasets.map(self.tokenize_function, batched=True)


        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset["train"],
            eval_dataset=tokenized_test_dataset["test"],
            compute_metrics=compute_metrics
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        trainer.evaluate()