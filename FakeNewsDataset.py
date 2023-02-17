# Do i need a dataset?
import os
import pandas as pd
from torch.utils.data import Dataset
import ast
from PIL import Image

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


class FakeNewsDataset(Dataset):
    def __init__(self, tsv_file, image_dir):
        self.data = pd.read_csv(tsv_file, sep='\t')
        self.img_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        id = row["ID"]
        type = row["Type"]
        text = row["Text"]
        label = row["Label"]

        img_paths = ast.literal_eval(row["Media"])
        images = []
        for img_path in img_paths:
            images.append(pil_loader(os.path.join(self.img_dir, img_path)))
        
        return {"id": id, "type": type, "text": text, "label": label, "images": images}

