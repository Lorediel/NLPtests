import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, AdamW, get_scheduler, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from NLPtests.utils import compute_metrics as cm
import numpy as np

class GilbertoModel():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("idb-ita/gilberto-uncased-from-camembert", do_lower_case=True)
        self.model = AutoModelForSequenceClassification.from_pretrained("idb-ita/gilberto-uncased-from-camembert", num_labels = 4)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    

    def tokenize_function(self, ds):
        return self.tokenizer(ds['Text'], truncation=True, max_length = 512)
    
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

    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return cm(predictions, labels)