from PIL import Image
import requests
import torch
import torch.nn as nn
import ast
from transformers import AdamW, get_scheduler, AutoProcessor, VisionTextDualEncoderModel, AutoTokenizer, AutoFeatureExtractor
import os
from tqdm.auto import tqdm
from NLPtests.utils import *
from NLPtests.FakeNewsDataset import collate_fn

labels_for_classification =  ["certainly a fake news", 
                              "probably a fake news", 
                              "probably a real news",
                              "certainly a real news"]



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.base_model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        self.processor = AutoProcessor.from_pretrained("clip-italian/clip-italian")
        self.tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear = nn.Linear(512*2, 4) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, pixel_values, nums_images):
        
        t_embeddings = self.base_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

        i_embeddings = self.base_model.get_image_features(pixel_values = pixel_values)

        #compute the max of the emnbeddings
        embeddings_images = []
        base = 0
        for i in range(len(nums_images)):
            tensor = i_embeddings[base:base+nums_images[i]]
            max_tensor, _ = torch.max(tensor, dim=0, keepdim=True)
            embeddings_images.append(max_tensor)
            base += nums_images[i]
        
        embeddings_images = torch.cat(embeddings_images, dim=0)
        embeddings = torch.cat((t_embeddings, embeddings_images), dim=1)
        logits = self.linear(embeddings)
        probs = self.softmax(logits)
        return logits, probs



        

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



    def train(self, train_ds, lr = 5e-5, num_epochs = 3, warmup_steps = 0, save_path = None):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=8, collate_fn = collate_fn
        )

        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
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
        for epoch in range(num_epochs):
            for batch in dataloader:
                #batch = {k: v.to(device) for k, v in batch.items()}
                texts = batch["text"]
                images_list = batch["images"]
                mask = batch["images_mask"]
                labels = batch["label"]

                # mask and flatten the list of images 
                nums_images = []
                for m in mask:
                    nums_images.append(sum(m))
                images_list = [item for sublist, mask_sublist in zip(images_list, mask)
                          for item, mask_value in zip(sublist, mask_sublist) 
                          if mask_value]

                
                t_inputs = self.model.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                i_inputs = self.model.processor(images = images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)
                labels = torch.tensor(labels).to(device)
                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)
                outputs = self.model(
                    input_ids=t_inputs.input_ids,
                    attention_mask=t_inputs.attention_mask,
                    pixel_values=i_inputs.pixel_values,
                    nums_images = nums_images,
                )
                
                logits = outputs[0]
                
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                metrics = compute_metrics(preds, batch["label"])
                print(metrics)
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