from PIL import Image
import requests
import torch
import torch.nn as nn
import ast
from transformers import AdamW, get_scheduler, AutoProcessor, VisionTextDualEncoderModel, AutoTokenizer, AutoFeatureExtractor, PretrainedConfig, PreTrainedModel
import os
from tqdm.auto import tqdm
from NLPtests.utils import *
from NLPtests.FakeNewsDataset import collate_fn



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

    def eval(self, ds):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=8, collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        with torch.no_grad():
            for batch in dataloader:
                texts = batch["text"]
                images_list = batch["images"]
                mask = batch["images_mask"]
                labels = batch["label"]

                nums_images = []
                for m in mask:
                    nums_images.append(sum(m))
                images_list = [item.to(device) for sublist, mask_sublist in zip(images_list, mask)
                          for item, mask_value in zip(sublist, mask_sublist) 
                          if mask_value]
                
                t_inputs = self.model.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                i_inputs = self.model.processor(images = images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)
                
                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)
                logits, probs = self.model(
                    input_ids=t_inputs.input_ids,
                    attention_mask=t_inputs.attention_mask,
                    pixel_values=i_inputs.pixel_values,
                    nums_images = nums_images,
                )

                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                total_preds += list(preds)
                total_labels += list(labels)
        metrics = compute_metrics(total_labels, total_preds)
        return metrics



    def train(self, train_ds, eval_ds, lr = 5e-5, num_epochs = 3, warmup_steps = 0, num_eval_steps = 10, save_path = "./"):
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
        current_step = 0
        best_eval = -1
        for epoch in range(num_epochs):
            for batch in dataloader:
                current_step += 1
                #batch = {k: v.to(device) for k, v in batch.items()}
                texts = batch["text"]
                images_list = batch["images"]
                mask = batch["images_mask"]
                labels = batch["label"]

                nums_images = []
                for m in mask:
                    nums_images.append(sum(m))
                images_list = [item.to(device) for sublist, mask_sublist in zip(images_list, mask)
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
                loss = criterion(logits, labels)
                if (current_step % num_eval_steps == 0):
                    eval_metrics = self.eval(eval_ds)
                    f1_score = eval_metrics["f1"]
                    if f1_score > best_eval:
                        torch.save(self.model.state_dict(), os.path.join(save_path, "best_model"))
                    self.model.train()
                

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                
        
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