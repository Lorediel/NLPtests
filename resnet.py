
from transformers import AutoImageProcessor, ResNetModel, ResNetForImageClassification, AdamW, get_scheduler
from tqdm.auto import tqdm
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torch.utils.data import Dataset
import os
import ast
import math


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


class ResnetModel:

    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels=4, ignore_mismatched_sizes=True)
        
    def test(self, datasets, dir_name):

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        eval_dataloader = torch.utils.data.DataLoader(
            datasets["valid"], batch_size=8
        )
        self.model.eval()
        total_predictions = []
        ground_truth = []
        for batch in eval_dataloader:
            with torch.no_grad():
                batch = {k: v for k, v in batch.items()}
                image_paths = batch["Media"]
                
                images = []
                images_per_row = []
                for row_paths in image_paths:
                  number_of_row_images = 0
                  for image_path in ast.literal_eval(row_paths): 
                    number_of_row_images += 1
                    image = pil_loader(os.path.join(dir_name, image_path))
                    image = self.image_processor(image, return_tensors="pt")
                    images.append(image["pixel_values"])
                  images_per_row.append(number_of_row_images)

                inputs = {}
                inputs["pixel_values"] = torch.cat(images, 0)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs.logits
                # compute the mean of the logits across the images of each row
                logits_per_row = []
                for number_of_row_images in images_per_row:
                    logits_per_row.append(torch.mean(logits[:number_of_row_images], dim=0))
                    logits = logits[number_of_row_images:]
                logits = torch.stack(logits_per_row, dim=0)

                predictions = torch.argmax(logits, dim=-1).to(device).type(torch.float)
                
                total_predictions+=predictions
                ground_truth+=batch["Label"].tolist()
        return total_predictions, ground_truth
                
    def train(self, datasets, dir_name, lr = 5e-5, num_epochs = 3, warmup_steps = 0):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        train_dataloader = torch.utils.data.DataLoader(
            datasets["train"], batch_size=8
        )
        self.model.train()
        # Initialize the optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps=len(train_dataloader) * num_epochs
        criterion = torch.nn.CrossEntropyLoss()
        # Initialize the scheduler
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v for k, v in batch.items()}
                image_paths = batch["Media"]
                
                images = []
                images_per_row = []
                for row_paths in image_paths:
                  number_of_row_images = 0
                  for image_path in ast.literal_eval(row_paths): 
                    number_of_row_images += 1
                    image = pil_loader(os.path.join(dir_name, image_path))
                    image = self.image_processor(image, return_tensors="pt")
                    images.append(image["pixel_values"])
                  images_per_row.append(number_of_row_images)

                inputs = {}
                inputs["pixel_values"] = torch.cat(images, 0)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                # compute the mean of the logits across the images of each row
                logits_per_row = []
                for number_of_row_images in images_per_row:
                    logits_per_row.append(torch.mean(logits[:number_of_row_images], dim=0))
                    logits = logits[number_of_row_images:]
                logits = torch.stack(logits_per_row, dim=0)
                labels = batch["Label"].to(device)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        return self.model
                

