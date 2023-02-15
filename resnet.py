
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

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


class ResnetModel:

    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        
    def test(self, datasets):
        eval_dataloader = torch.utils.data.DataLoader(
            datasets["valid"], batch_size=8
        )
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                #image_paths = batch["Media"]
                print(batch)
                """
                for image_path in image_paths:
                    image = pil_loader(image_path)
                    
                outputs = self.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                print(predictions)
                """


        
        
