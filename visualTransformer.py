from transformers import AutoImageProcessor, ViTForImageClassification, AdamW, get_scheduler
import torch
from tqdm.auto import tqdm
from NLPtests.utils import *
from NLPtests.FakeNewsDataset import collate_fn

class VisualTransformer():

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", num_labels=4)
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    def train(self, train_ds, val_ds, num_epochs= 3, lr = 5e-5,  warmup_steps = 0, batch_size = 8):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.train()
        self.model.to(device)
        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=8, shuffle=True, collate_fn = collate_fn
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
        for epoch in range(num_epochs):
            for batch in dataloader:
                current_step += 1
                #batch = {k: v.to(device) for k, v in batch.items()}
               
                images_list = batch["images"]
                mask = batch["images_mask"]
                labels = batch["label"]

                nums_images = []
                for m in mask:
                    nums_images.append(sum(m))
                images_list = [item.to(device) for sublist, mask_sublist in zip(images_list, mask)
                          for item, mask_value in zip(sublist, mask_sublist) 
                          if mask_value]

                inputs = self.processor(images = images_list, return_tensors="pt")
                print(inputs)