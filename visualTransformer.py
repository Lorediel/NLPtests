from transformers import AutoImageProcessor, ViTModel, AdamW, get_scheduler
import torch
from tqdm.auto import tqdm
from NLPtests.utils import *
from NLPtests.FakeNewsDataset import collate_fn
from math import floor
from NLPtests.utils import *
import os
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 4)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, pixel_values, nums_images):
        
        i_embeddings = self.base_model(pixel_values = pixel_values).pooler_output
        #computehe max of the emnbeddings
        embeddings_images = []
        base = 0
        
        # group the embeddings given by the model to take the max pooling
        for i in range(len(nums_images)):
            tensor = i_embeddings[base:base+nums_images[i]]
            max_tensor, _ = torch.max(tensor, dim=0, keepdim=True)
            embeddings_images.append(max_tensor)
            base += nums_images[i]
        
        embeddings_images = torch.cat(embeddings_images, dim=0)

        embeddings = self.linear1(embeddings_images)
        embeddings = self.relu(embeddings)
        embeddings = self.linear2(embeddings)
        embeddings = self.relu(embeddings)
        logits = self.linear3(embeddings)
        
        probs = self.softmax(logits)
        return logits, probs




class VisualTransformer():

    def __init__(self):
        self.model = Model()

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

    def train(self, train_ds, val_ds, num_epochs= 3, lr = 5e-5,  warmup_steps = 0, batch_size = 8, num_eval_steps = 10, save_path = "./"):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.train()
        self.model.to(device)
        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
        )
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
        criterion = nn.CrossEntropyLoss()
        best_metric = 0
        for epoch in range(num_epochs):
            for batch in dataloader:
                #current_step += 1
                #batch = {k: v.to(device) for k, v in batch.items()}
               
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]

                inputs = self.model.processor(images = images_list, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                
                labels_tensor = torch.tensor(labels).to(device)
                
                outputs = self.model(inputs["pixel_values"], nums_images)

                logits = outputs[0]

                loss = criterion(logits, labels_tensor)


                

                
                if (current_step % num_eval_steps == 0):
                    print("Epoch: ", epoch)
                    print("Loss: ", loss.item())
                    eval_metrics = self.eval(val_ds)
                    print("Eval metrics: ", eval_metrics)
                    f1_score = eval_metrics["f1"]
                    if f1_score > best_metric:
                        print("New best model found")
                        best_metric = f1_score
                        torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
                    print("Best metric: ", best_metric)
                    self.model.train()
                
                current_step += 1
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        
        return self.model, best_metric
        
