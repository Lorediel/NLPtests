import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_scheduler, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from NLPtests.utils import compute_metrics as cm
import numpy as np

class BertModel():

    checkpoint = "dbmdz/bert-base-italian-xxl-cased"
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.tokenizerLast = AutoTokenizer.from_pretrained(self.checkpoint, padding_side = 'left', truncation_side = 'left')
        self.model = BertForSequenceClassification.from_pretrained(self.checkpoint, num_labels = 4)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    

    def tokenize_function(self, ds):
        return self.tokenizer(ds['Text'], truncation=True, padding=True)
    
    def tokenizeLast_function(self, ds):
        return self.tokenizerLast(ds['Text'], truncation=True, padding=True)
    
    def tokenizeHeadTail_function(self, ds):
        texts = ds['Text']
        max_len = 512
        tokens = self.tokenizer(texts)
        half_len = int(max_len/2)
        post_tokens = {
            "input_ids": [],
            "attention_mask": []
        }
        for token_list in tokens.input_ids:
            new_tokens = []
            tl = len(token_list)
            if tl>max_len:
                new_tokens = token_list[:half_len] + token_list[-half_len:]
                attention_mask = [1] * max_len
            elif tl<=max_len:
                new_tokens = token_list
                attention_mask = [1] * tl + [0] * (max_len - tl)
            post_tokens["input_ids"].append(new_tokens)
            post_tokens["attention_mask"].append(attention_mask)
        return post_tokens
            
    def process_ds(self, datasets, tokenization_strategy = "first"):
        # Tokenize the datasets
        if tokenization_strategy == 'first':
            tokenized_ds = datasets.map(self.tokenize_function, batched=True)
        elif tokenization_strategy == 'last':
            tokenized_ds = datasets.map(self.tokenizeLast_function, batched=True)
        elif tokenization_strategy == 'head-tail':
            tokenized_ds = datasets.map(self.tokenizeHeadTail_function, batched=True)
        else:
            raise ValueError("tokenization_strategy must be either 'first', 'last' or 'head-tail'")

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

    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return cm(predictions, labels)