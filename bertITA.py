import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_scheduler, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from utils import *

class BertModel():

    checkpoint = "dbmdz/bert-base-italian-xxl-cased"
    def __init__(self, num_labels, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = BertForSequenceClassification.from_pretrained(self.checkpoint, num_labels = 4)

    def collate_fn_bert(self, batch):
        
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        return {'input_ids': torch.tensor(encodings['input_ids']), 'attention_mask': torch.tensor(encodings['attention_mask']), 'labels': torch.tensor(labels)}
    
    def train(self, train_dataset, eval_dataset, num_epochs = 3, lr = 5e-5, scheduler_type = "linear", warmup_steps = 0, batch_size = 8, logging_steps = 10):

        

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
            train_dataset = train_dataset,
            eval_dataset= eval_dataset,
            data_collator=self.collate_fn_bert,
            compute_metrics=compute_metrics
        )

        trainer.train()

    
