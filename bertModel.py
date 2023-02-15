from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from NLPtests.utils import build_dataloaders
from tqdm.auto import tqdm
import torch
import numpy as np

class BertModel:
    checkpoint = 'bert-base-uncased'


    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=4)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
    
    def tokenize_function(self, ds):
        return self.tokenizer(ds['Text'], truncation=True)
    
    def process_ds(self, datasets):
        # Tokenize the datasets
        tokenized_ds = datasets.map(self.tokenize_function, batched=True)

        # Rename the columns
        tokenized_ds = tokenized_ds.rename_column("Label", "labels")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.tokenized_ds = tokenized_ds
        return tokenized_ds

    def train(self, num_epochs = 3, lr = 5e-5, scheduler_type = "linear", warmup_steps = 0, batch_size = 8, eval_every_step = False):
            train_dataloader, eval_dataloader = build_dataloaders(self.tokenized_ds, self.data_collator)
            
            # Initialize the optimizer
            optimizer = AdamW(self.model.parameters(), lr=lr)
            num_training_steps=len(train_dataloader) * num_epochs
            # Initialize the scheduler
            if scheduler_type == "linear":
                scheduler = get_scheduler(
                    name=scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif scheduler_type == "cosine":
                scheduler = get_scheduler(
                    name=scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif scheduler_type == "cosine_with_restarts":
                scheduler = get_scheduler(
                    name=scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                    num_cycles=0.5
                )
            elif scheduler_type == "polynomial":
                scheduler = get_scheduler(
                    name=scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                    power=1.0
                )
            elif scheduler_type == "constant":
                scheduler = get_scheduler(
                    name=scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif scheduler_type == "constant_with_warmup":
                scheduler = get_scheduler(
                    name=scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            else:
                raise ValueError("Invalid scheduler type")
            

            # Start the training loop

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            training_args = TrainingArguments("test-trainer", evaluation_strategy="steps", logging_steps=10)
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset = self.tokenized_ds['train'],
                eval_dataset= self.tokenized_ds['valid'],
                data_collator=self.data_collator,
                compute_metrics=compute_metrics
            )

            trainer.train()
            
            

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}


"""
    def train(self, num_epochs = 3, lr = 5e-5, scheduler_type = "linear", warmup_steps = 0, batch_size = 8, eval_every_step = False):
        train_dataloader, eval_dataloader, test_dataloader = build_dataloaders(self.tokenized_ds, self.data_collator)
        
        # Initialize the optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps=len(train_dataloader) * num_epochs
        # Initialize the scheduler
        if scheduler_type == "linear":
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == "cosine":
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=0.5
            )
        elif scheduler_type == "polynomial":
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0
            )
        elif scheduler_type == "constant":
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == "constant_with_warmup":
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError("Invalid scheduler type")
        

        # Start the training loop

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        progress_bar = tqdm(range(num_training_steps))
        self.model.to(device)
        train_losses = []
        train_accuracies = []
        eval_accuracies = []
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                self.model.train()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).to(device)
                correct = torch.sum(predictions == batch["labels"])
                train_accuracy = correct / len(predictions)
                train_accuracies.append(train_accuracy)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                scheduler.step()
                optimizer.zero_grad()
                if (eval_every_step == True):
                    # eval on the validation set and compute the accuracy
                    self.model.eval()
                    total = 0
                    eval_correct = 0
                    for eval_batch in eval_dataloader:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        with torch.no_grad():
                            outputs_eval = self.model(**eval_batch)
                            logits = outputs_eval.logits
                            predictions = torch.argmax(logits, dim=-1).to(device)
                            eval_correct += torch.sum(predictions == eval_batch["labels"])
                            total += len(predictions)
                    eval_accuracy = eval_correct / total
                    eval_accuracies.append(eval_accuracy)
                progress_bar.update(1)

        return train_losses, train_accuracies, eval_accuracies

"""