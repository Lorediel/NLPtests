from PIL import Image
import requests
import torch
import torch.nn as nn
import ast
from transformers import CLIPProcessor, CLIPModel,AdamW, get_scheduler
import os
from tqdm.auto import tqdm

labels_for_classification =  ["certainly a fake news", 
                              "probably a fake news", 
                              "probably a real news",
                              "certainly a real news"]



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.base_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.linear = nn.Linear(768*2, 4) 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, text, images, device):
        inputs = self.processor(text=text, images= images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # You write you new head here
        outputs = self.base_model(**inputs)
        num_of_images = len(images)
        logits = []
        for i in range(num_of_images):
            single_outputs = torch.cat((outputs.image_embeds[i], outputs.text_embeds[0]), 0)
            out = self.linear(single_outputs)
            logits.append(out)
        #compute the mean of the logits
        logits = torch.mean(torch.stack(logits), dim=0)

        probs = self.softmax(logits)
        
        return {"logits":logits, "probs": probs}


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")

        
class ClipModel:

    def __init__(self):
        self.model = Model()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def test(self, datasets, dir_name):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.test()
        eval_dataloader = torch.utils.data.DataLoader(
            datasets["valid"], batch_size=8
        )
        progress_bar = tqdm(range(len(eval_dataloader)))
        for batch in eval_dataloader:
            batch = {k: v for k, v in batch.items()}
            image_paths = batch["Media"]
            
            images = []
            images_per_row = []
            for row_paths in image_paths:
                number_of_row_images = 0
                for image_path in ast.literal_eval(row_paths): 
                    number_of_row_images += 1
                    image = pil_loader(os.path.join(dir_name, image_path))
                    images.append(image)
                    images_per_row.append(number_of_row_images)
            text = batch["Text"]



    def train(self, datasets, dir_name, lr = 5e-5, num_epochs = 3, warmup_steps = 0, save_path = None):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        train_dataloader = torch.utils.data.DataLoader(
            datasets["train"], batch_size=8
        )
        self.model.train()
        # Initialize the optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps=len(train_dataloader) * num_epochs
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
                    images.append(image)
                  images_per_row.append(number_of_row_images)
                text = batch["Text"]
                outputs = self.model(text, images, device)
                logits = outputs["logits"]
                probs = outputs["probs"]
                labels = batch["Label"].to(device)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        if save_path is not None:
            self.model.save_pretrained(save_path)
        return self.model
                

"""
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
"""