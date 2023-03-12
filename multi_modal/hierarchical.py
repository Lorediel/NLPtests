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
import random
from NLPtests.focal_loss import FocalLoss

def post_process_logits(logits):
    log_fake = logits[:,0]
    log_real = logits[:,1]
    log_cert_fake = logits[:,2]
    log_prob_fake = logits[:,3]
    log_prob_real = logits[:,4]
    log_cert_real = logits[:,5]


    new_log_cert_fake = torch.min(log_fake, log_cert_fake)
    new_log_prob_fake = torch.min(log_fake, log_prob_fake)
    new_log_prob_real = torch.min(log_real, log_prob_real)
    new_log_cert_real = torch.min(log_real, log_cert_real)



    new_log_fake = torch.max(torch.max(log_fake, log_cert_fake), log_prob_fake)
    new_log_real = torch.max(torch.max(log_real, log_cert_real), log_prob_real)

    new_logits = torch.stack((new_log_fake, new_log_real, new_log_cert_fake, new_log_prob_fake,new_log_prob_real, new_log_cert_real), dim=1)

    return new_logits

def post_process_labels(labels):
    # 0: fake, 1: real, 2: cert_fake, 3: prob_fake, 4: prob_real, 5: cert_real
    label_dict = {
        0: [1,0,1,0,0,0],
        1: [1,0,0,1,0,0],
        2: [0,1,0,0,1,0],
        3: [0,1,0,0,0,1]
    }
    new_labels = []
    for label in labels:
        new_labels.append(label_dict[label])
    return torch.tensor(new_labels)

def logits_for_inference(logits):
    # Inference done by taking the max of the sum of the branches
    log_fake = logits[:,0]
    log_real = logits[:,1]
    log_cert_fake = logits[:,2]
    log_prob_fake = logits[:,3]
    log_prob_real = logits[:,4]
    log_cert_real = logits[:,5]

    branch_cert_fake = log_fake + log_cert_fake
    branch_prob_fake = log_fake + log_prob_fake
    branch_prob_real = log_real + log_prob_real
    branch_cert_real = log_real + log_cert_real

    final_logits = torch.stack((branch_cert_fake, branch_prob_fake, branch_prob_real, branch_cert_real), dim=1)
    return final_logits

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.base_model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        self.processor = AutoProcessor.from_pretrained("clip-italian/clip-italian")
        self.tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")
        self.tokenizerLast = AutoTokenizer.from_pretrained("clip-italian/clip-italian", padding_side = 'left', truncation_side = 'left')
        
        #self.dropout2 = nn.Dropout(0.2)
        #self.layernorm1 = nn.LayerNorm(512*2)
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 6)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, pixel_values):
        
        t_embeddings = self.base_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

        i_embeddings = self.base_model.get_image_features(pixel_values = pixel_values)

        #compute the max of the emnbeddings
        """
        embeddings_images = []
        base = 0
        for i in range(len(nums_images)):
            tensor = i_embeddings[base:base+nums_images[i]]
            max_tensor, _ = torch.max(tensor, dim=0, keepdim=True)
            embeddings_images.append(max_tensor)
            base += nums_images[i]
        """
        
        #embeddings_images = torch.cat(embeddings_images, dim=0)

        embeddings = torch.cat((t_embeddings, i_embeddings), dim=1)
        embeddings = self.relu(embeddings)
        #embeddings = self.layernorm1(embeddings)
        #embeddings = self.dropout2(embeddings)
        #embeddings = self.relu(embeddings)

        embeddings = self.linear1(embeddings)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(embeddings)


        logits = self.linear3(embeddings)

        logits = post_process_logits(logits)
        probs = self.softmax(logits)

        return logits, probs



        

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")
    

    
class ClipModel:
    #tokenizer_max_length = 512
    

    def __init__(self):
        self.model = Model()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def get_tokens(self, texts, tokenization_strategy):
        if tokenization_strategy == "first":
            return self.model.tokenizer(texts, return_tensors="pt", padding = True, truncation=True)
        elif tokenization_strategy == "last":
            return self.model.tokenizerLast(texts, return_tensors="pt", padding = True, truncation=True)
        elif tokenization_strategy == "head-tail":
            max_len = 512
            tokens = self.model.tokenizer(texts)
            half_len = int(max_len/2)
            post_tokens = {
                "input_ids": [],
                "attention_mask": []
            }
            max_token_len = 0
            for token_list in tokens.input_ids:
                tl = len(token_list)
                if tl>max_token_len:
                    max_token_len = tl
            max_len = min(max_token_len, max_len)
            for token_list in tokens.input_ids:
                new_tokens = []
                tl = len(token_list)
                if tl>max_len:
                    new_tokens = token_list[:half_len] + token_list[-half_len:]
                    new_tokens[-1] = 103
                    attention_mask = [1] * max_len
                elif tl<=max_len:
                    # add padding
                    new_tokens = token_list + [0] * (max_len - tl)
                    attention_mask = [1] * tl + [0] * (max_len - tl)
                post_tokens["input_ids"].append(new_tokens)
                post_tokens["attention_mask"].append(attention_mask)
            post_tokens["input_ids"] = torch.tensor(post_tokens["input_ids"])
            post_tokens["attention_mask"] = torch.tensor(post_tokens["attention_mask"])
            return post_tokens
        else:
            raise ValueError(f"tokenization_strategy {tokenization_strategy} not supported")

    def eval(self, ds, tokenization_strategy = "first", batch_size=8, print_confusion_matrix=False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        progress_bar = tqdm(range(len(dataloader)))
        with torch.no_grad():
            for batch in dataloader:
                texts = batch["text"]
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]

                random_images_list = []
                base = 0
                for i in range(len(nums_images)):
                    if nums_images[i] == 1:
                        random_images_list.append(images_list[base])
                        base += nums_images[i]
                        continue
                    random_index = random.randint(0, nums_images[i]-1)
                    sublist = images_list[base:base+nums_images[i]]
                    random_images_list.append(sublist[random_index])
                    base += nums_images[i]
                        
                
                #t_inputs = self.model.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                t_inputs = self.get_tokens(texts, tokenization_strategy)
                i_inputs = self.model.processor(images = random_images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)
                
                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)
                logits, probs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"],
                    pixel_values=i_inputs.pixel_values
                )
                inference_logits = logits_for_inference(logits).to(device)
                preds = torch.argmax(inference_logits, dim=1).detach().cpu().numpy()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        if print_confusion_matrix:
            display_confusion_matrix(total_preds, total_labels)
        return metrics


    def train(self, train_ds, eval_ds, lr = 5e-5, batch_size= 8, num_epochs = 3, warmup_steps = 0, save_path = "./", tokenization_strategy = "first"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
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
        # save the best model
        best_metrics = [0, 0, 0, 0, 0]
        print("accuracy | precision | recall | f1 | f1_weighted | f1_for_each_class")
        for epoch in range(num_epochs):
            for batch in dataloader:
                current_step += 1
                #batch = {k: v.to(device) for k, v in batch.items()}
                texts = batch["text"]
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]


                random_images_list = []
                base = 0
                for i in range(len(nums_images)):
                    if nums_images[i] == 1:
                        random_images_list.append(images_list[base])
                        base += nums_images[i]
                        continue
                    random_index = random.randint(0, nums_images[i]-1)
                    sublist = images_list[base:base+nums_images[i]]
                    random_images_list.append(sublist[random_index])
                    base += nums_images[i]

                t_inputs = self.get_tokens(texts, tokenization_strategy)
                i_inputs = self.model.processor(images = random_images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)

                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)
                outputs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"],
                    pixel_values=i_inputs.pixel_values
                )
                
                logits = outputs[0]

                new_logits = post_process_logits(logits).to(device)
                new_labels = post_process_labels(labels).to(device).float()
                
                
                loss = criterion(new_logits, new_labels)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(eval_ds, tokenization_strategy, batch_size=batch_size)
            print("Eval metrics: ", format_metrics(eval_metrics))
            f1_score = eval_metrics["f1_weighted"]
            if f1_score > min(best_metrics):
                best_metrics.remove(min(best_metrics))
                best_metrics.append(f1_score)
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
            print("Best metrics: ", best_metrics)
            self.model.train()
        
        return best_metrics
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        return self.model
                
