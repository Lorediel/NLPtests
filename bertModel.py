from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from torch.utils.data import DataLoader
from NLPtests.utils import build_dataloaders
from tqdm.auto import tqdm
import torch

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
            
    def train(self, num_epochs = 3, lr = 5e-5, scheduler_type = "linear", warmup_steps = 0):
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
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
