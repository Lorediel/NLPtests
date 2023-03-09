from PIL import Image
import requests
import torch
import torch.nn as nn
import ast
from transformers import AdamW, get_scheduler, AutoModel, AutoProcessor, VisionTextDualEncoderModel, AutoTokenizer, AutoFeatureExtractor, PretrainedConfig, PreTrainedModel
import os
from tqdm.auto import tqdm
from NLPtests.utils import *
from NLPtests.FakeNewsDataset import collate_fn

class BertParts(nn.Module):

  def __init__(self, pretrained_path = None):
    super().__init__()
    self.bert = AutoModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    if (pretrained_path != None):
        s = torch.load(pretrained_path)
        new_s = {}
        for n in s:
            if (n.startswith("bert")):
                new_s[n[5:]] = s[n]
        self.bert.load_state_dict(new_s)
    self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    self.max_len = 512
    for param in self.bert.parameters():
       param.requires_grad = False
    self.pooler = nn.Sequential(
      nn.Linear(768, 768),
      nn.Tanh(),
    )
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def forward(self, texts):
    tokens_list = self.tokenizer(texts).input_ids    
    output = []
    masks = []
    num_sublists = []
    divided_tokens = []
    longest = 0
    for tokens in tokens_list:
      tokens.pop(0) #remove cls
      tokens.pop(-1) #remove eos
      n = 0
      
      for x in range(0, len(tokens), self.max_len - 2):
        chunk = [102] + tokens[x:x+self.max_len-2] + [103]
        mask = [1] * self.max_len
        #pad the last chunk
        if (len(chunk) != self.max_len):
          mask = [1] * len(chunk) + [0] * (self.max_len - len(chunk))
          chunk = chunk + ([0] * (self.max_len - len(chunk)))
        divided_tokens.append(chunk)   
        masks.append(mask)
        n+=1
      num_sublists.append(n)
    
    input_ids = torch.tensor(divided_tokens).to(self.device)
    attention_masks = torch.tensor(masks).to(self.device)
    bertOutput = self.bert(input_ids, attention_masks).last_hidden_state


    base = 0
    final = []
    for i in range(len(num_sublists)):
      tensors = bertOutput[base:base+num_sublists[i]]
      mean_tensor = torch.mean(tensors, dim = 0)[0]
      mean_tensor = self.pooler(mean_tensor)

      final.append(mean_tensor)
      base += num_sublists[i]
    final = torch.stack(final, dim=0).to(self.device)
    
    return final

class Model(nn.Module):
    def __init__(self, pretrained_path = None):
        super(Model, self).__init__()
        
        self.base_model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        self.processor = AutoProcessor.from_pretrained("clip-italian/clip-italian")
        self.bert = BertParts(pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.tokenizerLast = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased", padding_side = 'left', truncation_side = 'left')
        
        self.tanh = nn.Tanh()
        #self.dropout2 = nn.Dropout(0.2)
        #self.layernorm1 = nn.LayerNorm(512*2)

        self.linear1 = nn.Linear(1280, 1280)
        #self.linear2 = nn.Linear(1280, 1280)
        self.linear3 = nn.Linear(1280, 4)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(1280)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, texts, pixel_values):
        
        t_embeddings = self.bert(texts)

        i_embeddings = self.base_model.get_image_features(pixel_values = pixel_values)
        embeddings_images = i_embeddings
        #embeddings_images = torch.cat(embeddings_images, dim=0)
        # Using tanh because the pooler output of bert is tanh
        embeddings_images = self.tanh(embeddings_images)
        embeddings = torch.cat((t_embeddings, embeddings_images), dim=1)

        #embeddings = self.layernorm1(embeddings)
        #embeddings = self.dropout2(embeddings)
        #embeddings = self.relu(embeddings)

        embeddings = self.linear1(embeddings)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(embeddings)

        #embeddings = self.linear2(embeddings)
        #embeddings = self.layernorm(embeddings)
        #embeddings = self.dropout(embeddings)
        #embeddings = self.relu(embeddings)

        logits = self.linear3(embeddings)
        probs = self.softmax(logits)

        return logits, probs



        

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")
    

    
class ClipBertModel:
    

    def __init__(self, pretrained_path = None):
        self.model = Model(pretrained_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def eval(self, ds, tokenization_strategy = "first", batch_size=8):
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
                

                i_inputs = self.model.processor(images = random_images_list, return_tensors="pt", padding=True)

                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)


                logits, probs = self.model(
                    texts = texts,
                    pixel_values=i_inputs.pixel_values
                )

                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_labels, total_preds)
        return metrics



    def train(self, train_ds, eval_ds, lr = 5e-5, batch_size= 8, num_epochs = 3, warmup_steps = 0, num_eval_steps = 10, save_path = "./", tokenization_strategy = "first"):
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
        best_metric = 0
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
                

                i_inputs = self.model.processor(images = random_images_list, return_tensors="pt", padding=True)

                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)

                logits, probs = self.model(
                    texts = texts,
                    pixel_values=i_inputs.pixel_values
                )
                
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                loss = criterion(logits, torch.tensor(labels).to(device))
                

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            print("Epoch: ", epoch)
            print("Loss: ", loss.item())
            eval_metrics = self.eval(eval_ds, tokenization_strategy, batch_size=batch_size)
            print("Eval metrics: ", eval_metrics)
            f1_score = eval_metrics["f1"]
            if f1_score > best_metric:
                print("New best model found")
                best_metric = f1_score
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print("Best metric: ", best_metric)
            self.model.train()
        
        return self.model
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        return self.model
                
