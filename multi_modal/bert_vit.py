
import torch
import torch.nn as nn
from transformers import ResNetModel, BertModel, AutoTokenizer, AutoImageProcessor, AdamW, get_scheduler, ViTModel
from tqdm.auto import tqdm
from NLPtests.utils import compute_metrics, format_metrics
from NLPtests.FakeNewsDataset import collate_fn
import os

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.bert = BertModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.flatten = nn.Flatten(1,-1)
        self.linear1 = nn.Linear(1536, 1536)
        self.layer_norm = nn.LayerNorm(1536)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(1536, 1536)
        self.layer_norm2 = nn.LayerNorm(1536)
        self.linear3 = nn.Linear(1536, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, pixel_values, nums_images):
        
        # Image embeddings extraction from vit
        i_embeddings = self.vit(pixel_values = pixel_values).pooler_output
        
        #compute the max of the embeddings
        embeddings_images = []
        base = 0
        for i in range(len(nums_images)):
            tensor = i_embeddings[base:base+nums_images[i]]
            max_tensor, _ = torch.max(tensor, dim=0, keepdim=True)
            embeddings_images.append(max_tensor)
            base += nums_images[i]
        
        embeddings_images = torch.cat(embeddings_images, dim=0)

        #textual embeddings extraction from bert
        embeddings_text = self.bert(input_ids = input_ids, attention_mask = attention_mask).pooler_output

        concatenated_tensor = torch.cat((embeddings_images, embeddings_text), dim=1)

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

class BertVitConcatModel():

    def __init__(self):
        self.model = Model()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    

    def eval(self, ds, batch_size = 8):
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

                
                t_inputs = self.model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                i_inputs = self.model.processor(images_list, return_tensors="pt", padding=True)

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

                preds = torch.argmax(logits, dim=1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
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
        best_metrics = [0, 0, 0, 0, 0]
        print("accuracy | precision | recall | f1 | f1_weighted | f1_for_each_class")
        for epoch in range(num_epochs):
            for batch in dataloader:
                texts = batch["text"]
                images_list = batch["images"]
                labels = batch["label"]

                nums_images = batch["nums_images"]

                t_inputs = self.model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                i_inputs = self.model.processor(images_list, return_tensors="pt", padding=True)

                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)

                labels_tensor = torch.tensor(labels).to(device)

                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)
                logits, probs = self.model(
                    input_ids=t_inputs.input_ids,
                    attention_mask=t_inputs.attention_mask,
                    pixel_values=i_inputs.pixel_values,
                    nums_images = nums_images,
                )

                loss = criterion(logits, labels_tensor)



                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_step += 1
                progress_bar.update(1)
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(val_ds)
            print("Eval metrics: ", format_metrics(eval_metrics))
            f1_score = eval_metrics["f1_weighted"]
            if f1_score > min(best_metrics):
                best_metrics.remove(min(best_metrics))
                best_metrics.append(f1_score)
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
            print("Best metrics: ", best_metrics)
            self.model.train()

        return self.model