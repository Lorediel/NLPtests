import torch
from transformers import AdamW, get_scheduler, AutoProcessor, VisionTextDualEncoderModel, AutoTokenizer, BertModel, AdamW, get_scheduler
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from NLPtests.FakeNewsDataset import collate_fn
import torch.nn as nn
from tqdm.auto import tqdm
import os
import numpy as np
from NLPtests.utils import compute_metrics, format_metrics
import random



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Clip

       
        self.clip = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        self.processor = AutoProcessor.from_pretrained("clip-italian/clip-italian")
        self.clipTokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")
        self.clipTokenizerLast = AutoTokenizer.from_pretrained("clip-italian/clip-italian", padding_side = 'left', truncation_side = 'left')

        # image captioning
        self.vit = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.captionTokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # bert
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bertTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bertTokenizerLast = AutoTokenizer.from_pretrained("bert-base-uncased", padding_side = 'left', truncation_side = 'left')


        self.relu = nn.ReLU()
        self.linear1 = nn.Sequential(
            nn.Linear(1792, 1792),
            nn.LayerNorm(1792),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(1792, 1792),
            nn.LayerNorm(1792),
            nn.Dropout(0.1),
            nn.ReLU(),
        )
        self.linear4 = nn.Linear(1792, 4)

        self.softmax = nn.Softmax(dim=1)
            

        
    def forward(self, images, input_ids, attention_mask, pixel_values):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # clip embeddings
        t_embeddings = self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

        i_embeddings = self.clip.get_image_features(pixel_values = pixel_values) 

        clip_embeddings = torch.cat((t_embeddings, i_embeddings), dim=1) # clip_embeddings.shape = (batch_size, 1024)
 
        
        # Take the image captions 
        with torch.no_grad():
            pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            
            output_ids = self.vit.generate(pixel_values=pixel_values)
            preds = self.captionTokenizer.batch_decode(output_ids, skip_special_tokens=True)
            captions = [pred.strip() for pred in preds]

        t_inputs = self.bertTokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        input_ids = t_inputs.input_ids.to(device)
        attention_mask = t_inputs.attention_mask.to(device)
        caption_embeddings = self.bert(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state[:,0,:] # caption_embeddings.shape = (batch_size, 768)
        


        # Concatenate the embeddings
        final_embeddings = torch.cat((clip_embeddings, caption_embeddings), dim=1) # final_embeddings.shape = (batch_size, 2816)
        

        final_embeddings = self.relu(final_embeddings)
        final_embeddings = self.linear1(final_embeddings)
        #final_embeddings = self.linear2(final_embeddings)
        final_embeddings = self.linear3(final_embeddings)
        final_embeddings = self.linear4(final_embeddings)
        logits = final_embeddings
        probs = self.softmax(logits)
        return logits, probs




class Concatenated_Model():
    def __init__(self):
        self.model = Model()

    def get_tokens_clip(self, texts, tokenization_strategy):
        if tokenization_strategy == "first":
            return self.model.clipTokenizer(texts, return_tensors="pt", padding = True, truncation=True)
        elif tokenization_strategy == "last":
            return self.model.clipTokenizerLast(texts, return_tensors="pt", padding = True, truncation=True)
        else:
            raise ValueError(f"tokenization_strategy {tokenization_strategy} not supported")

        
    def eval(self, ds,tokenization_strategy="first", batch_size = 8 ):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
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
                t_inputs = self.get_tokens_clip(texts, tokenization_strategy)
                i_inputs = self.model.processor(images = random_images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)

                outputs = self.model(
                    images = random_images_list,
                    input_ids = t_inputs.input_ids,
                    attention_mask = t_inputs.attention_mask,
                    pixel_values = i_inputs.pixel_values
                )

                logits = outputs[0]

                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        return metrics
                

    
    def train(self, train_ds, eval_ds, batch_size = 8, num_epochs = 3, num_eval_steps = 10,lr = 1e-5, warmup_steps = 0, tokenization_strategy="first", save_path = "./"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                t_inputs = self.get_tokens_clip(texts, tokenization_strategy)
                i_inputs = self.model.processor(images = random_images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)
                
                outputs = self.model(
                    images = random_images_list,
                    input_ids = t_inputs.input_ids,
                    attention_mask = t_inputs.attention_mask,
                    pixel_values = i_inputs.pixel_values
                )
                
                
                
                logits = outputs[0]
                labels = torch.tensor(labels).to(device)
                loss = criterion(logits, labels)
                

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                current_step += 1
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(eval_ds, tokenization_strategy= tokenization_strategy, batch_size=batch_size)
            print("Eval metrics: ", format_metrics(eval_metrics))
            f1_score = eval_metrics["f1_weighted"]
            if f1_score > min(best_metrics):
                best_metrics.remove(min(best_metrics))
                best_metrics.append(f1_score)
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
            print("Best metrics: ", best_metrics)
            self.model.train()
        return best_metrics