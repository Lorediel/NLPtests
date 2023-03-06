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

    def __init__(self, d_model: int = 768, nhead: int = 4, d_hid: int = 768,
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

  def __init__(self):
    super().__init__()
    self.bert = AutoModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    self.max_len = self.tokenizer.model_max_length
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def forward(self, texts):
    tokens_list = self.tokenizer(texts).input_ids.to(self.device)
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

    bertOutput = self.bert(input_ids, attention_masks).last_hidden_state[:,0,:]
    max_pad = max(num_sublists)
    out = []
    masks = []
    base = 0
    masks = []
    outSize = bertOutput.shape[1]

    for i in range(len(num_sublists)):
      tensors = bertOutput[base:base+num_sublists[i]]
      mask = [0.] * num_sublists[i] + [1.] * (max_pad - num_sublists[i])
      mask_tensor = torch.tensor(mask)
      if num_sublists[i] != max_pad:
        # pad the tensors
        pad = torch.zeros(max_pad - num_sublists[i], outSize)
        tensors = torch.cat((tensors, pad), dim=0)
      out.append(tensors)
      masks.append(mask_tensor)
    out = torch.stack(out, dim=0)
    final_mask = torch.stack(masks, dim=0).bool()
    return {"out": out, "mask": final_mask}
    

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertParts()
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

           
        

    def forward(self, input_ids, attention_mask, only_cls = False):
        self.bert.eval()
        with torch.no_grad():
            bert_output = self.bert(input_ids)
        # take only the cls
        cls_out = self.attention(bert_output["out"].transpose(0,1), bert_output["mask"]).transpose(0,1)[:,0,:] # [batch, 768]
        cls_out = self.relu(cls_out)

        cls_out = self.linear1(cls_out)
        cls_out = self.linear2(cls_out)
        cls_out = self.linear3(cls_out)
        logits = cls_out
        return logits

    

class LongBert():

    def __init__(self):
        self.model = Model()
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
