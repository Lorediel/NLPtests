from transformers import ResNetModel, BertModel, AutoModel,  AutoTokenizer, AutoImageProcessor, AdamW, get_scheduler
from tqdm.auto import tqdm
from NLPtests.utils import compute_metrics
from NLPtests.FakeNewsDataset import collate_fn
import os

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


import math
from typing import Tuple

class h_transformer(nn.Module):

    def __init__(self, d_model: int = 768, nhead: int = 8, d_hid: int = 768,
                 nlayers: int = 6, dropout: float = 0.25):

        # PARAMETERS
        # d_model:  length of the tokens
        # nhead:    # of heads in the attention layer
        # d_hid:    dimension of the dense layer
        # n_layers: # of layers

        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # TransformerEncoderLayer is a transformer layer made up of self attention + feed forward network
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # TransformerEncoder is a stack of N layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.d_model = d_model


    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask = mask)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BertParts(nn.Module):

  def __init__(self, pretrained_model_path = None):
    super().__init__()
    self.bert = AutoModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    if (pretrained_model_path != None):
        s = torch.load(pretrained_model_path)
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
    
    input_ids = torch.tensor(divided_tokens)
    attention_masks = torch.tensor(masks)

    bertOutput = self.bert(input_ids, attention_masks).pooler_output
    
    base = 0
    final = []
    for i in range(len(num_sublists)):
      tensors = bertOutput[base:base+num_sublists[i]]
      mean_tensor = torch.mean(tensors, dim = 0)
      final.append(mean_tensor)
      base += num_sublists[i]
    final = torch.stack(final, dim=0)
    
    return final
    

class Model(nn.Module):
    def __init__(self, pretrained_model_path = None):
        super(Model, self).__init__()
        self.bertParts = BertParts(pretrained_model_path)
        self.attention = h_transformer()
        self.relu = nn.ReLU()
        self.linear1 = nn.Sequential(
          nn.Linear(768, 768),
          nn.LayerNorm(768),
          nn.Dropout(0.1),
          nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
          nn.Linear(768, 768),
          nn.LayerNorm(768),
          nn.Dropout(0.1),
          nn.ReLU(),
        )
        self.linear3 = nn.Linear(768, 4)

           
    def forward(self, texts):
        bert_output = self.bertParts(texts)
        # take only the cls
        #cls_out = self.attention(bert_output["out"].transpose(0,1), bert_output["mask"]).transpose(0,1)[:,0,:] # [batch, 768]
        #cls_out = self.relu(bert_output)

        cls_out = self.linear1(bert_output)
        #cls_out = self.linear2(cls_out)
        cls_out = self.linear3(cls_out)
        logits = cls_out
        return logits

    

class LongBert():

    def __init__(self, pretrained_model_path = None):
        self.model = Model(pretrained_model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def eval(self, ds, batch_size = 8):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
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
                labels = batch["label"]
                
                
                logits = self.model(texts)

                preds = torch.argmax(logits, dim=1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_labels, total_preds)
        return metrics

    def train(self, train_ds, val_ds, lr = 5e-5, batch_size= 8, num_epochs = 25, save_path = "./"):
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
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        current_step = 0
        # save the best model
        best_metrics = [0, 0, 0]
        b_metrics = 0
        for epoch in range(num_epochs):
            for batch in dataloader:
                texts = batch["text"]
                labels = batch["label"]

                labels_tensor = torch.tensor(labels).to(device)

                logits = self.model(texts)

                loss = criterion(logits, labels_tensor)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_step += 1
                progress_bar.update(1)
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(val_ds, batch_size = batch_size)
            print("Eval metrics: ", eval_metrics)
            f1_score = eval_metrics["f1_weighted"]
            if f1_score > min(best_metrics):
                best_metrics.remove(min(best_metrics))
                best_metrics.append(f1_score)
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
            print("Best metric (f1_weighted): ", best_metrics)
            self.model.train()
        return b_metrics