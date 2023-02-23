from transformers import AutoImageProcessor, ResNetModel, AdamW, get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import ast
from PIL import Image
from NLPtests.FakeNewsDataset import collate_fn
from tqdm.auto import tqdm
from NLPtests.utils import compute_metrics

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.base_model = ResNetModel.from_pretrained("microsoft/resnet-18")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        self.flatten = nn.Flatten(1,-1)
        self.linear = nn.Linear(512, 4)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, pixel_values, nums_images):
        
        
        i_embeddings = self.base_model(pixel_values = pixel_values).pooler_output
        i_embeddings = self.flatten(i_embeddings)
        #compute the max of the embeddings
        embeddings_images = []
        base = 0
        for i in range(len(nums_images)):
            tensor = i_embeddings[base:base+nums_images[i]]
            max_tensor, _ = torch.max(tensor, dim=0, keepdim=True)
            embeddings_images.append(max_tensor)
            base += nums_images[i]
        
        embeddings_images = torch.cat(embeddings_images, dim=0)
       
        
        logits = self.linear(embeddings_images)
        probs = self.softmax(logits)
        return logits, probs


class ResnetModel():
    def __init__(self):
        self.model = Model()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def eval(self, ds, batch_size = 8):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        progress_bar = tqdm(range(len(dataloader)))
    
        with torch.no_grad():
            for batch in dataloader:
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]
                
                
                inputs = self.model.processor(images = images_list, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                
                
                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)
                logits, _ = self.model(inputs["pixel_values"], nums_images)
                preds = logits.argmax(dim=-1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_labels, total_preds)
        return metrics

    
    def train(self, train_ds, val_ds, lr = 5e-5, batch_size= 8, num_epochs = 3, warmup_steps = 0, num_eval_steps = 10, save_path = "./"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.train()
        self.model.to(device)

        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
        )

        criterion = nn.CrossEntropyLoss()
        
        
        # Initialize the optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps=len(dataloader) * num_epochs
        # Initialize the scheduler
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        current_step = 0
        # save the best model
        best_metric = 0
        for epoch in range(num_epochs):
            for batch in dataloader:
                
                images_list = batch["images"]
                nums_images = batch["nums_images"]
                labels = batch["label"]



                i_inputs = self.model.processor(images = images_list, return_tensors="pt")

                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)

                labels = torch.tensor(labels).to(device)

                
                outputs = self.model(
                    pixel_values=i_inputs["pixel_values"],
                    nums_images = nums_images,
                )

                logits = outputs[0]
                loss = criterion(logits, labels)

                best_metric = 0
                if (current_step % num_eval_steps == 0):
                    print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
                    eval_metrics = self.eval(val_ds)
                    print("Eval metrics: ", eval_metrics)
                    f1_score = eval_metrics["f1"]
                    if f1_score > best_metric:
                        print("New best model found")
                        best_metric = f1_score
                        torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
                    self.model.train()
                
                current_step += 1
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)