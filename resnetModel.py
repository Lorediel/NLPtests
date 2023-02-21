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
        
        self.base_model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.flatten = nn.Flatten(1,-1)
        self.linear = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 4)
        self.relu = nn.ReLU()
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
       
        
        layer1 = self.linear(embeddings_images)
        layer1 = self.relu(layer1)
        layer2 = self.linear2(layer1)
        layer2 = self.relu(layer2)
        logits = self.linear3(layer2)
        probs = self.softmax(logits)
        return logits, probs


class ResnetModel():
    def __init__(self):
        self.model = Model()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    
    def train(self, train_ds, eval_ds, lr = 5e-5, batch_size= 8, num_epochs = 3, warmup_steps = 0, num_eval_steps = 10, save_path = "./"):
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
                #current_step += 1
                images_list = batch["images"]
                mask = batch["images_mask"]
                labels = batch["label"]

                nums_images = []
                for m in mask:
                    nums_images.append(sum(m))
                """
                new_l = []
                for j in range(len(mask)):
                    for i in range(len(mask[j])):
                        if mask[j][i] == 1:
                            new_l.append(images_list[j][i])
                images_list = new_l
                """
                images_list = [item.to(device) for sublist, mask_sublist in zip(images_list, mask)
                          for item, mask_value in zip(sublist, mask_sublist) 
                          if mask_value]
                

                i_inputs = self.model.processor(images = images_list, return_tensors="pt")

                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)

                labels = torch.tensor(labels).to(device)

                
                outputs = self.model(
                    pixel_values=i_inputs["pixel_values"],
                    nums_images = nums_images,
                )

                logits = outputs[0]
                preds = logits.argmax(dim=-1).tolist()
                loss = criterion(logits, labels)

                metrics = compute_metrics(preds, batch["label"])
                print(metrics)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)