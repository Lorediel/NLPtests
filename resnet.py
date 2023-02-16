
from transformers import AutoImageProcessor, ResNetModel
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torch.utils.data import Dataset
import os

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


class ResnetModel:

    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        
    def test(self, datasets, dir_name):
        eval_dataloader = torch.utils.data.DataLoader(
            datasets["valid"], batch_size=8
        )
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                image_paths = batch["Media"]
                
                images = []
                for image_path in image_paths:
                    image = pil_loader(os.path.join(dir_name, image_path))
                    image = self.image_processor(image, return_tensors="pt")
                    images.append(image)

                print(images)
                """
                outputs = self.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                print(predictions)
                """