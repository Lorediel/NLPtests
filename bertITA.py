import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_scheduler, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from NLPtests.utils import *

class BertModel():

    checkpoint = "dbmdz/bert-base-italian-xxl-cased"
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = BertForSequenceClassification.from_pretrained(self.checkpoint, num_labels = 4)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    

    def tokenize_function(self, ds):
        return self.tokenizer(ds['Text'], truncation=True)
    
    def process_ds(self, datasets):
        # Tokenize the datasets
        tokenized_ds = datasets.map(self.tokenize_function, batched=True)

        # Rename the columns
        tokenized_ds = tokenized_ds.rename_column("Label", "labels")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        #self.tokenized_ds = tokenized_ds
        return tokenized_ds


    def train(self, dataset, num_epochs = 3, lr = 5e-5, scheduler_type = "linear", warmup_steps = 0, batch_size = 8, logging_steps = 10):



        training_args = TrainingArguments("test-trainer",
         evaluation_strategy="steps",
          logging_steps=logging_steps,
          per_device_train_batch_size=batch_size,
          per_device_eval_batch_size=batch_size,
          lr_scheduler_type=scheduler_type,
          warmup_steps=warmup_steps,
          num_train_epochs=num_epochs,
          learning_rate=lr,
          )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset = dataset['train'],
            eval_dataset= dataset['valid'],
            data_collator=self.data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()

    
