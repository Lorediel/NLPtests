
import torch
import torch.nn as nn
from transformers import ResNetModel, BertModel, AutoTokenizer, AutoImageProcessor, AdamW, get_scheduler
from tqdm.auto import tqdm
from NLPtests.utils import compute_metrics
from NLPtests.FakeNewsDataset import collate_fn
import os
from NLPtests.focal_loss import FocalLoss
import random

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-18")
        self.bert = BertModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.tokenizerLast = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased", padding_side = 'left', truncation_side = 'left')
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        self.flatten = nn.Flatten(1,-1)
        self.linear1 = nn.Linear(768 + 512, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(512, 512)
        self.layer_norm2 = nn.LayerNorm(512)
        self.linear3 = nn.Linear(512, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, pixel_values):
        
        # Image embeddings extraction from resnet
        i_embeddings = self.resnet(pixel_values = pixel_values).pooler_output
        
        

        #textual embeddings extraction from bert
        embeddings_text = self.bert(input_ids = input_ids, attention_mask = attention_mask).pooler_output

        concatenated_tensor = torch.cat((i_embeddings, embeddings_text), dim=1)

        #embeddings = self.relu(concatenated_tensor)

        embeddings = self.linear1(concatenated_tensor)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(embeddings)

        embeddings = self.linear2(embeddings)
        embeddings = self.layer_norm2(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(embeddings)
        
        logits = self.linear3(embeddings)
        
        probs = self.softmax(logits)
        return logits, probs

class BertResnetConcatModel():

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

    def eval(self, ds, tokenization_strategy, batch_size = 8):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size,shuffle=True, collate_fn = collate_fn
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

                t_inputs = self.get_tokens(texts, tokenization_strategy)
                i_inputs = self.model.processor(random_images_list, return_tensors="pt", padding=True).to(device)

                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)

                logits, probs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"],
                    pixel_values=i_inputs.pixel_values
                )

                preds = torch.argmax(logits, dim=1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_labels, total_preds)
        return metrics

    def train(self, train_ds, val_ds, lr = 5e-5, focal_loss = False, batch_size= 8, num_epochs = 3, warmup_steps = 0,save_path = "./", tokenization_strategy = "first"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.train()
        self.model.to(device)

        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
        )

        if focal_loss:
            criterion = FocalLoss(gamma=2, reduction='sum')
        else:
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
                i_inputs = self.model.processor(random_images_list, return_tensors="pt", padding=True).to(device)

                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)

                labels_tensor = torch.tensor(labels).to(device)
                logits, probs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"],
                    pixel_values=i_inputs.pixel_values
                )

                loss = criterion(logits, labels_tensor)



                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_step += 1
                progress_bar.update(1)
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(val_ds, tokenization_strategy, batch_size = batch_size)
            print("Eval metrics: ", eval_metrics)
            f1_score = eval_metrics["f1"]
            if f1_score > best_metric:
                print("New best model found")
                best_metric = f1_score
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print("Best metric: ", best_metric)
            self.model.train()
        return self.model