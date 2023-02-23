import os
import pandas as pd
from torch.utils.data import Dataset, Subset
import ast
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


class FakeNewsDataset(Dataset):
    def __init__(self, tsv_file, image_dir):
        self.data = pd.read_csv(tsv_file, sep='\t')
        self.img_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor()
        ])

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
            #images.append(img_path)
            image = pil_loader(os.path.join(self.img_dir, img_path))
            transformed_image = self.transform(image)
            images.append(transformed_image)
        
        return {"id": id, "type": type, "text": text, "label": label, "images": images}
        

def collate_fn(batch):
    ids = []
    types = []
    texts = []
    labels = []
    images = []
    max_images = 0
    for sample in batch:
        images_list = sample["images"]
        #find the maximum number of images in a batch
        num_images = len(images_list)
        if num_images > max_images:
            max_images = num_images
        ids.append(sample["id"])
        types.append(sample["type"])
        texts.append(sample["text"])
        labels.append(sample["label"])
    
    nums = []
    for sample in batch:
        images_list = sample["images"]
        num_images = len(images_list)
        nums.append(num_images)
        #pad the images list with a tensor of zeros
        for i in range(num_images, max_images):
            images_list.append(torch.zeros(3, 336, 336))
        
        images.append(images_list)
    mask = []
    for n in nums:
        image_mask = [1] * n + [0] * (max_images - n)
        mask.append(image_mask)
    
    
    return {"id": ids, "type": types, "text": texts, "label": labels, "images_mask": mask, "images": images}


if __name__ == "__main__":
    ds = FakeNewsDataset("/Users/lorenzodamico/Documents/Uni/tesi/NLPtests/MULTI-Fake-Detective_Task1_Data.tsv", "/Users/lorenzodamico/Documents/Uni/tesi/content/Media")
    
    dataloader = DataLoader(
        ds, batch_size=8, shuffle=True, collate_fn = collate_fn
    )
    for batch in dataloader:
        print(batch)
        break
    