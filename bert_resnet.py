
import torch
import torch.nn as nn
from transformers import ResNetModel, BertModel, AutoTokenizer, AutoImageProcessor, AdamW, get_scheduler
from tqdm.auto import tqdm
from NLPtests.utils import compute_metrics
from FakeNewsDataset import collate_fn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-18")
        self.bert = BertModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        self.flatten = nn.Flatten(1,-1)
        self.linear = nn.Linear(768 + 512, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, pixel_values, nums_images):
        
        # Image embeddings extraction from resnet
        i_embeddings = self.resnet(pixel_values = pixel_values).pooler_output
        i_embeddings = self.flatten(i_embeddings)
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

        logits = self.linear(concatenated_tensor)
        probs = self.softmax(logits)
        return logits, probs

class BertResnetConcatModel():

    def __init__(self):
        self.model = Model()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    

    def eval(self, ds, batch_size = 8):
        pass

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
        best_metric = 0
        for epoch in range(num_epochs):
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
                preds = torch.argmax(probs, dim=1)
                metrics = compute_metrics(preds, labels)
                print(metrics)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_step += 1
                progress_bar.update(1)

        return self.model