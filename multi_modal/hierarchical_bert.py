
import torch
import torch.nn as nn
from transformers import ResNetModel, BertModel, AutoModel,  AutoTokenizer, AutoImageProcessor, AdamW, get_scheduler
from tqdm.auto import tqdm
from NLPtests.utils import compute_metrics, format_metrics, display_confusion_matrix
from NLPtests.FakeNewsDataset import collate_fn
import os
from NLPtests.focal_loss import FocalLoss
from NLPtests.hierarchical_segmentation_head import SegmentationHead

def post_process_labels(labels):
    # 0: fake, 1: real, 2: cert_fake, 3: prob_fake, 4: prob_real, 5: cert_real
    label_dict = {
        0: [1,0,0,0],
        1: [0,1,0,0],
        2: [0,0,1,0],
        3: [0,0,0,1],
    }
    new_labels = []
    for label in labels:
        new_labels.append(label_dict[label])
    return torch.tensor(new_labels)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.tokenizerLast = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased", padding_side = 'left', truncation_side = 'left')
        self.segmentation_head = SegmentationHead(768)
        self.softmax = nn.Softmax(dim=1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, input_ids, attention_mask):
        
        embeddings_text = self.bert(input_ids = input_ids, attention_mask = attention_mask).pooler_output

        fake_array, all_logits = self.segmentation_head(embeddings_text) # logits = [batch_size, 2]

        final_probabilities = []

        for i in range(len(fake_array)):
            if fake_array[i] == "fake":
                fake_prob = self.softmax(all_logits[i])
                final_probabilities += [[0,0] +fake_prob[0].tolist()]
            else:
                real_prob = self.softmax(all_logits[i])
                final_probabilities += [real_prob[0].tolist() + [0,0]]

        final_probabilities = torch.tensor(final_probabilities).to(self.device)
        return final_probabilities
    
class BertModel():

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
        
    
    def eval(self, ds, tokenization_strategy, batch_size = 8, print_confusion_matrix = False):
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
                labels = batch["label"]


                
                t_inputs = self.get_tokens(texts, tokenization_strategy)

                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                
                probs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"]
                )

                preds = torch.argmax(probs, dim=1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        if print_confusion_matrix:
            display_confusion_matrix(total_preds, total_labels)
        return metrics
    
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        return self.model

    def train(self, train_ds, val_ds, lr = 5e-5, batch_size= 8, num_epochs = 3, warmup_steps = 0, num_eval_steps = 10, save_path = "./", tokenization_strategy = "first"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.train()
        self.model.to(device)

        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
        )
        criterion = nn.BCELoss()
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
                labels = batch["label"]


                t_inputs = self.get_tokens(texts, tokenization_strategy)

                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)

                labels_tensor = post_process_labels(labels).to(device)

                probs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"]
                )

                loss = criterion(probs, labels_tensor)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_step += 1
                progress_bar.update(1)
            
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(val_ds, tokenization_strategy, batch_size = batch_size)
            print("Eval metrics: ", format_metrics(eval_metrics))
            f1_score = eval_metrics["f1_weighted"]
            if f1_score > min(best_metrics):
                best_metrics.remove(min(best_metrics))
                best_metrics.append(f1_score)
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
            print("Best metrics: ", best_metrics)
            self.model.train()
        return self.model
    
    